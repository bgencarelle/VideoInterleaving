#!/usr/bin/env python3
"""Measure an audio loop through a cable, mixer or tape monitor at nominal speed.

Connect the selected outputs through the path under test to the selected input
device. Each output sends a timing sweep then a 1 kHz tone separately. Results
are digital dBFS and relative path gain, not calibrated analog dBu.
"""
import argparse
import json
from pathlib import Path
import sys
import threading

import numpy as np
from scipy.signal import chirp, correlate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem.audio_common import device, pair, sounddevice


def db(value):
    return float(20*np.log10(value)) if value > 0 else None


def make_probe(rate, outputs=(0, 1), level_dbfs=-18., max_delay=1.):
    if not np.isfinite(rate) or rate < 8000:
        raise ValueError('The level check requires a sample rate of at least 8 kHz')
    if not np.isfinite(level_dbfs) or not -60 <= level_dbfs <= -3:
        raise ValueError('--level-dbfs must be between -60 and -3 (peak level)')
    if not np.isfinite(max_delay) or not .05 <= max_delay <= 10:
        raise ValueError('--max-delay-ms must be between 50 and 10000')
    amplitude = 10**(level_dbfs/20)
    pilot_count, tone_count = round(.15*rate), round(.4*rate)
    t = np.arange(pilot_count)/rate
    pilot = amplitude*chirp(t, f0=300, f1=min(7000, .4*rate), t1=t[-1])*np.hanning(pilot_count)
    tone = amplitude*np.sin(2*np.pi*1000*np.arange(tone_count)/rate)
    ramp = max(1, round(.01*rate))
    tone[:ramp] *= np.linspace(0, 1, ramp)
    tone[-ramp:] *= np.linspace(1, 0, ramp)
    cursor = round(.25*rate)
    slots = []
    for channel in outputs:
        slots.append(dict(channel=channel, pilot_start=cursor,
                          tone_start=cursor+pilot_count, tone_count=tone_count))
        cursor += pilot_count+tone_count+round((max_delay+.1)*rate)
    audio = np.zeros((cursor, max(outputs)+1), np.float32)
    for slot in slots:
        c, a, b = slot['channel'], slot['pilot_start'], slot['tone_start']
        audio[a:a+pilot_count,c] = pilot
        audio[b:b+tone_count,c] = tone
    return audio, pilot, slots


def analyze(recorded, rate, pilot, slots, level_dbfs, max_delay):
    recorded = np.asarray(recorded, dtype=float)
    if recorded.ndim != 2 or not np.isfinite(recorded).all():
        raise ValueError('Recording must contain finite multichannel samples')
    noise = np.sqrt(np.mean(recorded[:max(1,round(.15*rate))]**2, axis=0))
    reference_rms = 10**(level_dbfs/20)/np.sqrt(2)
    pilot_energy = float(np.einsum('i,i->', pilot, pilot, optimize=False))
    rows = []
    for slot in slots:
        start = slot['pilot_start']
        for channel in range(recorded.shape[1]):
            window = recorded[start:start+len(pilot)+round(max_delay*rate),channel]
            if len(window) < len(pilot):raise ValueError('Recording is too short')
            corr = correlate(window, pilot, mode='valid', method='fft')
            energy = np.concatenate(([0.], np.cumsum(window**2)))
            energy = np.maximum(energy[len(pilot):]-energy[:-len(pilot)],0)
            score = np.minimum(1., np.abs(corr)/np.sqrt(np.maximum(energy*pilot_energy,1e-30)))
            # Cumulative-energy subtraction loses precision after loud signals.
            # Do not interpret roundoff in a silent tail as a perfect match.
            floor = max(float(energy.max())*64*np.finfo(float).eps, 1e-25)
            score = np.where(energy > floor, score, 0.)
            lag = int(np.argmax(score))
            tone_start = slot['tone_start']+lag
            lo = tone_start+round(.08*rate)
            hi = tone_start+slot['tone_count']-round(.08*rate)
            tone = recorded[lo:hi,channel]
            if len(tone) != hi-lo:raise ValueError('Recording is too short for the returned tone')
            rms = float(np.sqrt(np.mean(tone**2)))
            peak = float(np.max(np.abs(tone)))
            detected = bool(score[lag] >= .2 and rms > max(2*noise[channel],1e-8))
            rows.append(dict(output=slot['channel']+1,input=channel+1,detected=detected,
                correlation=float(score[lag]),delay_ms=1000*lag/rate if detected else None,
                input_rms_dbfs=db(rms) if detected else None,
                input_peak_dbfs=db(peak) if detected else None,
                gain_db=db(rms/reference_rms) if detected else None,
                noise_rms_dbfs=db(noise[channel]),
                signal_to_idle_noise_db=db(rms/noise[channel]) if detected and noise[channel]>0 else None,
                near_full_scale_percent=100*float(np.mean(np.abs(tone)>=.999)) if detected else None))
    return rows


def run_capture(sd, sent, input_count, rate, input_device, output_device):
    """Callback only copies samples; measurements run after capture ends."""
    received = np.zeros((len(sent),input_count),np.float32)
    done = threading.Event()
    state = dict(position=0, input_overflows=0, output_underflows=0, errors=[])
    def callback(indata, outdata, frames, timing, status):
        outdata.fill(0)
        try:
            state['input_overflows'] += bool(status.input_overflow)
            state['output_underflows'] += bool(status.output_underflow)
            at = state['position']; count = min(frames,len(sent)-at)
            outdata[:count] = sent[at:at+count]
            received[at:at+count] = indata[:count]
            state['position'] += count
        except Exception as exc:
            state['errors'].append(exc)
            raise sd.CallbackAbort()
        if state['position'] >= len(sent):raise sd.CallbackStop()
    with sd.Stream(device=(input_device,output_device), samplerate=rate,
                   channels=(input_count,sent.shape[1]), dtype='float32', blocksize=0,
                   latency='high', callback=callback, finished_callback=done.set) as stream:
        if not done.wait(len(sent)/rate+10):
            stream.abort()
            raise RuntimeError('Audio check timed out')
    if state['errors']:raise RuntimeError(str(state['errors'][0])) from state['errors'][0]
    if state['position'] != len(sent):raise RuntimeError('Audio check ended before capture completed')
    return received, {k:state[k] for k in ('input_overflows','output_underflows')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-device',type=device)
    p.add_argument('--output-device',type=device)
    p.add_argument('--output-channels',type=pair,default=(0,1))
    p.add_argument('--level-dbfs',type=float,default=-18.,help='Transmit peak dBFS (default -18)')
    p.add_argument('--max-delay-ms',type=float,default=1000.)
    p.add_argument('--json',type=Path,help='Save all measured output/input paths')
    p.add_argument('--list-devices',action='store_true')
    args = p.parse_args(argv)
    sd = sounddevice()
    if args.list_devices:print(sd.query_devices());return
    source = sd.query_devices(args.input_device,'input')
    sink = sd.query_devices(args.output_device,'output')
    if source['hostapi'] != sink['hostapi']:
        p.error('Choose input and output entries using the same host API (e.g. both MME)')
    rate = float(source['default_samplerate'])
    channels = int(source['max_input_channels'])
    if channels < 1:p.error('Selected device has no input channels')
    if max(args.output_channels) >= sink['max_output_channels']:p.error('Output channel is unavailable')
    try:
        sent,pilot,slots = make_probe(rate,args.output_channels,args.level_dbfs,args.max_delay_ms/1000)
    except ValueError as exc:p.error(str(exc))
    print(f'{rate:g} Hz; transmit {args.level_dbfs:g} dBFS peak; '
          f'{len(sent)/rate:.1f} seconds; listening to {channels} inputs',flush=True)
    captured,flags = run_capture(sd,sent,channels,rate,args.input_device,args.output_device)
    rows = analyze(captured,rate,pilot,slots,args.level_dbfs,args.max_delay_ms/1000)
    result = dict(sample_rate=rate,output_peak_dbfs=args.level_dbfs,
                  measurement_valid=not any(flags.values()),**flags,paths=rows)
    for row in rows[:channels]:
        noise = row['noise_rms_dbfs']
        label = f'{noise:.1f} dBFS RMS' if noise is not None else 'digital silence'
        print(f"Input {row['input']}: idle noise {label}")
    print('OUT -> IN   DELAY ms   GAIN dB   RMS dBFS   PEAK dBFS   NEAR FULL SCALE %')
    for row in rows:
        if row['detected']:
            print(f"{row['output']:3} -> {row['input']:<3} {row['delay_ms']:9.2f} {row['gain_db']:9.2f} "
                  f"{row['input_rms_dbfs']:10.2f} {row['input_peak_dbfs']:11.2f} "
                  f"{row['near_full_scale_percent']:19.3f}")
    for slot in slots:
        paths = [r for r in rows if r['output']==slot['channel']+1 and r['detected']]
        if not paths:print(f"Output {slot['channel']+1}: no identifiable return within delay window")
        else:
            best = max(paths,key=lambda r:r['input_rms_dbfs'])
            print(f"Output {best['output']}: strongest return on input {best['input']}")
    if any(flags.values()):print(f'Capture had glitches; repeat before trusting levels or delay: {flags}')
    if args.json:
        args.json.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(f'Saved {args.json}')


if __name__ == '__main__':
    try:main()
    except KeyboardInterrupt:pass
