#!/usr/bin/env python3
"""Measure harmonic folding and packet recovery at the 2x/48 kHz boundary.

Run from the repository root::

    .venv/bin/python test_modem_v7/alias_fold.py

The tone probe synthesizes a 28 kHz component and its second and third
harmonics at 192 kHz, then compares naïve decimation with polyphase
anti-alias resampling to 48 kHz. The packet probe compares the same encoded V7
reverse series at 2x/96 kHz, 2x/48 kHz with the normal anti-alias resampler,
and 2x/48 kHz with deliberately naïve decimation. It opens no audio device.
"""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
from scipy.signal import butter, resample_poly, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7                                      # noqa: E402


TONE_RATE = 192_000
OUTPUT_RATE = 48_000
HARMONICS = ((28_000, 1.0), (56_000, .25), (84_000, .125))
PACKET_COUNTERS = (1, 2, 3, 4, 7, 8, 9, 10)


def folded_frequency(frequency, sample_rate):
    """Nyquist-fold a tone frequency into [0, sample_rate/2]."""
    remainder = float(frequency) % float(sample_rate)
    return min(remainder, float(sample_rate)-remainder)


def tone_amplitude(samples, sample_rate, frequency):
    """Single-bin sinusoid amplitude for the integer-cycle probe tones."""
    values = np.asarray(samples, dtype=np.float64)
    indexes = np.arange(len(values), dtype=np.float64)
    phase = 2*np.pi*float(frequency)*indexes/float(sample_rate)
    return float(2*np.abs(np.dot(values, np.exp(-1j*phase)))/len(values))


def run_tone_probe():
    count = TONE_RATE  # one second; every probe tone completes integer cycles
    time_axis = np.arange(count, dtype=np.float64)/TONE_RATE
    source = sum(amplitude*np.sin(2*np.pi*frequency*time_axis)
                 for frequency, amplitude in HARMONICS).astype(np.float32)
    naive = source[::TONE_RATE//OUTPUT_RATE]
    filtered = resample_poly(source, OUTPUT_RATE, TONE_RATE).astype(np.float32)

    measurements = []
    for frequency, source_amplitude in HARMONICS:
        alias = folded_frequency(frequency, OUTPUT_RATE)
        naive_amplitude = tone_amplitude(naive, OUTPUT_RATE, alias)
        filtered_amplitude = tone_amplitude(filtered, OUTPUT_RATE, alias)
        measurements.append({
            'source_hz': frequency,
            'source_amplitude': source_amplitude,
            'alias_hz_at_48k': alias,
            'naive_decimation_amplitude': naive_amplitude,
            'anti_aliased_amplitude': filtered_amplitude,
            'suppression_db': float(20*np.log10(
                max(filtered_amplitude, 1e-15)/
                max(naive_amplitude, 1e-15))),
        })
        if abs(naive_amplitude-source_amplitude) > 1e-4:
            raise AssertionError(f'{frequency} Hz did not fold to {alias} Hz')
        if filtered_amplitude > source_amplitude*.01:
            raise AssertionError(f'{frequency} Hz harmonic was not suppressed')
    return measurements


def _packet_series():
    target = .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10))
    model = v7.load_model(target, 'nearest')
    with Image.open(v7.REFERENCE_FIXTURE) as image:
        sources = (image.convert('RGB'),
                   ImageOps.mirror(image.convert('RGB')))
    values = [v7.image_values(v7.prepare_image(image, 'nearest'),
                              model.coder.grids, 'nearest')
              for image in sources]
    packets = [v7.encode_pulse_frame(
        model, values[index % 2], counter=counter, aspect_code=6,
        source_index=index, pilot_tones=True, eof_marker=True)
        for index, counter in enumerate(PACKET_COUNTERS)]
    return model, np.concatenate(packets).astype(np.float32)


def _decode_reverse(model, stream, rate):
    margin = int(np.ceil(v7.REVERSE_PACKET_MARGIN*rate/(v7.RATE*2)))+8
    silence = np.zeros((margin, 2), dtype=np.float32)
    capture = np.concatenate((silence, stream[::-1].copy(), silence))
    hits = v7.pulse_frame_hits(capture, sample_rate=rate,
                               direction='reverse')
    received, eof_count = [], 0
    for start, scale, _confidence, _direction in hits:
        frames, info = v7.decode_reverse_packet(
            model, capture, start, scale,
            state=v7.PulseState(tail_memory=False), sample_rate=rate,
            pilot_timing='tone-seeded')
        eof_count += int(info.get('eof_markers_validated') or 0)
        received.extend(int(frame.diag['source_index']) for frame in frames
                        if frame.status == 'received' and
                        frame.diag.get('metadata_valid') and
                        not frame.diag.get('metadata_provisional'))
    return {
        'capture_rate_hz': rate,
        'expected_pulse_scale': rate/(v7.RATE*2),
        'pulse_hits': len(hits),
        'eof_markers_validated': eof_count,
        'received_count': len(received),
        'source_indices': received,
    }


def run_packet_probe():
    model, wire = _packet_series()
    capture96 = v7.speed_pulse_stream(wire, 2, rate=96_000)
    # This is the production polyphase path for a 48 kHz output clock.
    capture48_aa = v7.speed_pulse_stream(wire, 2, rate=48_000)
    # Deliberately omit the resampler's anti-alias filter for comparison.
    capture48_naive = capture96[::2].copy()
    results = {
        '2x_96k': _decode_reverse(model, capture96, 96_000),
        '2x_48k_polyphase': _decode_reverse(model, capture48_aa, 48_000),
        '2x_48k_naive_decimation': _decode_reverse(model, capture48_naive,
                                                    48_000),
    }
    for cutoff in (13_000, 15_000, 18_000):
        tx_lpf = butter(4, cutoff, btype='lowpass', fs=v7.RATE, output='sos')
        wire_lpf = sosfiltfilt(tx_lpf, wire, axis=0).astype(np.float32)
        tag = f'{cutoff//1000}khz_sender_lpf'
        results[f'2x_96k_with_{tag}'] = _decode_reverse(
            model, v7.speed_pulse_stream(wire_lpf, 2, rate=96_000), 96_000)
        results[f'2x_48k_with_{tag}'] = _decode_reverse(
            model, v7.speed_pulse_stream(wire_lpf, 2, rate=48_000), 48_000)
    expected_order = list(reversed(range(len(PACKET_COUNTERS))))
    if results['2x_96k']['source_indices'] != expected_order:
        raise AssertionError('96 kHz 2x reference did not decode in source order')
    for key in ('2x_48k_polyphase', '2x_48k_naive_decimation',
                '2x_48k_with_13khz_sender_lpf',
                '2x_48k_with_15khz_sender_lpf',
                '2x_48k_with_18khz_sender_lpf'):
        row = results[key]
        if row['pulse_hits'] != len(PACKET_COUNTERS):
            raise AssertionError(f'{key} did not retain all pulse headers')
        if row['eof_markers_validated'] != len(PACKET_COUNTERS):
            raise AssertionError(f'{key} did not validate all EOF markers')
        if row['received_count'] >= len(PACKET_COUNTERS):
            raise AssertionError(f'{key} unexpectedly recovered every image')
    return results


def main():
    report = {
        'tone_harmonics': run_tone_probe(),
        'reverse_v7_packets': run_packet_probe(),
        'interpretation': (
            'Naive 48 kHz decimation folds out-of-band harmonics into the '
            'baseband. Polyphase conversion suppresses them, but the lost '
            'V7 carriers are not restored; 96 kHz retains the clean 2x wire.'),
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
