#!/usr/bin/env python3
"""Live Xvfb -> V7 Fold-500 -> PortAudio dropout relay -> live receiver.

Run from the repository root on PulseAudio/PipeWire-Pulse. Sender, duplex
PortAudio relay and GL receiver are live; the relay mutes only one leg for
seeded 1-20 ms bursts. Run separately with ``--side left`` and ``--side
right`` to test each intact leg, or use ``--side random`` to alternate.

Example: ``.venv/bin/python tools/v7_stereo_dropout_torture.py --side left``
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
RATE = 44100
BLOCK = 2048
WARMUP_SECONDS = 2.0
WARMUP_FRAMES = round(WARMUP_SECONDS * 12.243)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--side', choices=('none', 'left', 'right', 'random'),
                    default='random',
                    help='leg to mute; random selects a leg per burst')
parser.add_argument('--receiver-profile', choices=('adaptive', 'fold-500'),
                    default='adaptive',
                    help='exercise status dispatch or pin the known Fold-500 profile')
parser.add_argument('--seconds', type=float, default=15.0)
parser.add_argument('--seed', type=int, default=20260929)
args = parser.parse_args()
DURATION = args.seconds
SEED = args.seed
if DURATION < 3:
    parser.error('--seconds must be at least 3')
OUT = ROOT / f'tmp/v7-stereo-dropout-{args.side}-{time.strftime("%Y%m%d-%H%M%S")}'
OUT.mkdir(parents=True, exist_ok=True)
processes = []
files = []
source_blocks = []
relayed_blocks = []
status_events = []
rng = np.random.default_rng(SEED)
module_ids = []


def start(args, env, name):
    f = open(OUT / name, 'w')
    files.append(f)
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=f,
                         stderr=subprocess.STDOUT, text=True)
    processes.append(p)
    return p


def stop(p, sig=15):
    if p and p.poll() is None:
        p.send_signal(sig)
        try:
            p.wait(timeout=6)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def wait_for_log(p, path, marker, timeout=60):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if p.poll() is not None:
            raise RuntimeError(f'{path.name} process exited with {p.returncode}')
        if path.exists() and marker in path.read_text(errors='replace'):
            return
        time.sleep(.05)
    raise TimeoutError(f'timed out waiting for {marker!r} in {path}')


# Independent, reproducible bursts. A burst mutes only one side, with the
# requested 1-20 ms duration; the other stereo side is left untouched.
schedule = []
when = round((WARMUP_SECONDS + .35) * RATE)
while when < round((WARMUP_SECONDS + DURATION - .25) * RATE):
    ms = int(rng.integers(1, 21))
    count = round(ms * RATE / 1000)
    if args.side == 'none':
        break
    side = int(rng.integers(0, 2)) if args.side == 'random' else (
        {'left': 0, 'right': 1}[args.side])
    schedule.append({'start': when, 'samples': count, 'ms': ms, 'side': side})
    # Keep the events separated enough to isolate per-packet leg tolerance;
    # the independent leg choice still creates random opposite-leg sequences.
    when += round(float(rng.uniform(.14, .24)) * RATE)

xv_src = xv_view = clock = receiver = sender = None
try:
    for sink_name, description in (('v7_drop_src', 'V7DropSource'),
                                   ('v7_drop_dst', 'V7DropReceiver')):
        loaded = subprocess.run(
            ['pactl', 'load-module', 'module-null-sink',
             f'sink_name={sink_name}', f'rate={RATE}', 'channels=2',
             f'sink_properties=device.description={description}'],
            cwd=ROOT, check=True, capture_output=True, text=True)
        module_ids.append(loaded.stdout.strip())
    xv_src = start(['Xvfb', ':96', '-screen', '0', '1280x720x24',
                    '+extension', 'GLX', '-nolisten', 'tcp'], os.environ.copy(),
                   'xvfb-source.log')
    xv_view = start(['Xvfb', ':97', '-screen', '0', '1280x720x24',
                     '+extension', 'GLX', '-nolisten', 'tcp'], os.environ.copy(),
                    'xvfb-viewer.log')
    time.sleep(.5)
    src_env = os.environ.copy()
    src_env['DISPLAY'] = ':96'
    view_env = os.environ.copy()
    view_env.update(DISPLAY=':97', LIBGL_ALWAYS_SOFTWARE='1',
                    PULSE_SOURCE='v7_drop_dst.monitor')
    clock = start(['xclock', '-digital', '-update', '1',
                   '-geometry', '420x100+20+20'], src_env, 'xclock.log')
    receiver_args = ['.venv/bin/python', 'tools/v7_live.py', 'receive',
                     '--device', 'pulse', '--audio-muted',
                     '--no-sync-warning', '--diagnostics']
    if args.receiver_profile == 'fold-500':
        receiver_args += ['--experimental-fold', '500']
    receiver = start(receiver_args, view_env,
                     'receiver.log')
    wait_for_log(receiver, OUT / 'receiver.log', 'V7 receive ready')

    # Direct PortAudio duplex bridge: capture the sender's PulseAudio monitor,
    # independently zero scheduled slices in one channel, and write to the
    # receiver's PulseAudio sink. Sender/receiver remain the real CLI programs.
    os.environ.update(PULSE_SOURCE='v7_drop_src.monitor', PULSE_SINK='v7_drop_dst')
    cursor = 0
    event_index = 0
    active_origin = None
    applied_events = 0

    def callback(indata, outdata, frames, timing, status):
        global cursor, event_index, active_origin, applied_events
        if status:
            status_events.append(str(status))
        block = np.array(indata, dtype=np.float32, copy=True)
        source_blocks.append(block)
        # PulseAudio monitor noise can be above 1e-4 before the sender starts.
        # Anchor the burst clock to real modem-level audio, not that noise.
        if (active_origin is None and
                float(np.max(np.abs(block))) > 1e-2):
            active_origin = cursor
        out = block.copy()
        end = cursor + frames
        while (active_origin is not None and event_index < len(schedule) and
               active_origin+schedule[event_index]['start'] < end):
            event = schedule[event_index]
            event_start = active_origin+event['start']
            a = max(0, event_start - cursor)
            b = min(frames, event_start + event['samples'] - cursor)
            if b > a:
                out[a:b, event['side']] = 0
                applied_events += 1
            event_index += 1
        outdata[:] = out
        relayed_blocks.append(out.copy())
        cursor = end

    with sd.Stream(samplerate=RATE, blocksize=BLOCK, channels=2,
                   dtype='float32', device='pulse', latency='high',
                   callback=callback):
        sender_env = os.environ.copy()
        sender_env.update(DISPLAY=':96', PULSE_SINK='v7_drop_src')
        sender = start(['.venv/bin/python', 'tools/v7_live.py', 'send',
                        '--source', 'screen', '--screen-backend', 'mss',
                        '--device', 'pulse', '--profile', 'fold-500',
                        '--capture-fps', '12', '--seconds',
                        str(DURATION+WARMUP_SECONDS),
                        '--no-log'], sender_env, 'sender.log')
        sender.wait(timeout=max(30, DURATION+WARMUP_SECONDS+12))
        time.sleep(.5)

    captured = np.concatenate(source_blocks) if source_blocks else np.zeros((0, 2), np.float32)
    relayed = np.concatenate(relayed_blocks) if relayed_blocks else np.zeros((0, 2), np.float32)
    np.save(OUT / 'portaudio-source.npy', captured)
    np.save(OUT / 'portaudio-relayed.npy', relayed)
    (OUT / 'dropout-schedule.json').write_text(json.dumps(schedule, indent=2) + '\n')
    screen_env = os.environ.copy()
    screen_env['DISPLAY'] = ':97'
    subprocess.run(['.venv/bin/python', '-c',
                    'from PIL import ImageGrab; ImageGrab.grab().save(r"' +
                    str(OUT / 'receiver.png') + '")'], cwd=ROOT,
                   env=screen_env, check=True)
finally:
    stop(sender)
    stop(receiver, 2)
    stop(clock)
    stop(xv_src)
    stop(xv_view)
    for module_id in reversed(module_ids):
        subprocess.run(['pactl', 'unload-module', module_id], cwd=ROOT,
                       check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    for f in files:
        f.close()

# Include degraded displayable frames as well as clean frames; a loss must not
# disappear from the report simply because its status was not "received".
lines = (OUT / 'receiver.log').read_text(errors='replace').splitlines()
all_rows = []
for line in lines:
    if line.startswith('{'):
        try:
            item = ast.literal_eval(line)
        except Exception:
            continue
        if 'counter' in item:
            all_rows.append(item)
# Ignore acquisition warmup; the test interval begins only after the live
# receiver has established its packet clock and decoded startup probes.
rows = [row for row in all_rows if row.get('counter', 0) > WARMUP_FRAMES]
good = [r for r in rows if r.get('displayable')]
indices = [r.get('source_index') for r in good if r.get('source_index') is not None]
steps = [b - a for a, b in zip(indices, indices[1:])]
lost = sum(not bool(r.get('displayable')) for r in rows)
status_lost_displayable = sum(
    r.get('status') == 'lost' and r.get('displayable') for r in rows)
index_gaps = [(a, b) for a, b in zip(indices, indices[1:]) if b - a != 1]
erasure_symbols = [
    sum((r.get('stereo_erased_symbols') or [0, 0])[side] for r in rows)
    for side in (0, 1)]
sys.path.insert(0, str(ROOT))
from animation_modem import v7
raw = np.load(OUT / 'portaudio-source.npy')
relayed = np.load(OUT / 'portaudio-relayed.npy')
audio_origin = 0 if active_origin is None else active_origin
dropouts_on_signal = 0
for event in schedule:
    first = audio_origin + event['start']
    last = first + event['samples']
    segment = raw[max(0, first):min(len(raw), last), event['side']]
    if segment.size and float(np.max(np.abs(segment))) > 1e-3:
        dropouts_on_signal += 1
report = {
    'rate': RATE, 'duration_seconds': DURATION, 'seed': SEED,
    'receiver_profile': args.receiver_profile,
    'warmup_frames_ignored': len(all_rows)-len(rows),
    'captured_frames': len(captured), 'relay_status_events': status_events,
    'audio_origin_sample': active_origin,
    'dropouts_injected': applied_events,
    'dropouts_overlapping_modem_audio': dropouts_on_signal,
    'stereo_erased_symbols_by_leg': erasure_symbols,
    'dropouts': {'count': len(schedule),
                 'by_side': {'left': sum(d['side'] == 0 for d in schedule),
                             'right': sum(d['side'] == 1 for d in schedule)},
                  'duration_ms_min_max': ([min(d['ms'] for d in schedule),
                                           max(d['ms'] for d in schedule)]
                                          if schedule else [])},
    'receiver': {
        'displayable': len(good),
        'lost': lost,
        'status_lost_displayable': status_lost_displayable,
        'metadata_valid': sum(bool(r.get('metadata_valid')) for r in rows),
        'first_last_source_index': [indices[0], indices[-1]] if indices else [],
        'source_index_step_counts': {str(s): steps.count(s) for s in sorted(set(steps))},
        'source_index_gaps': index_gaps,
        'input_gaps': sum("'status': 'input_gap_reacquire'" in line for line in lines),
        'decoder_errors': sum('decoder_exception' in line for line in lines),
        'profile_switches': sum("'status': 'wire_profile_switch'" in line for line in lines),
    },
    'portaudio': {
        'source_packet_headers_by_leg': [
            len(v7.pulse_frame_starts(raw[:, side], sample_rate=RATE))
            for side in (0, 1)],
        'relayed_packet_headers_by_leg': [
            len(v7.pulse_frame_starts(relayed[:, side], sample_rate=RATE))
            for side in (0, 1)],
        'source_rms_by_leg': np.sqrt(np.mean(raw*raw, axis=0)).tolist(),
        'relayed_rms_by_leg': np.sqrt(np.mean(relayed*relayed, axis=0)).tolist(),
    },
}
intact_side = (1 if args.side == 'left' else 0) if args.side in ('left', 'right') else None
if intact_side is not None:
    report['portaudio']['intact_leg_header_loss'] = (
        report['portaudio']['source_packet_headers_by_leg'][intact_side] -
        report['portaudio']['relayed_packet_headers_by_leg'][intact_side])
if args.side == 'none':
    # A handful of pilot residual outliers occur on an otherwise clean live
    # capture; the control rejects broad/continuous false-erasure detection.
    erasures_match = all(count <= 12 for count in erasure_symbols)
elif args.side == 'left':
    erasures_match = (erasure_symbols[0] > 0 and
                      erasure_symbols[1] <= max(12, .02*erasure_symbols[0]))
elif args.side == 'right':
    erasures_match = (erasure_symbols[1] > 0 and
                      erasure_symbols[0] <= max(12, .02*erasure_symbols[1]))
else:
    erasures_match = (all(count > 0 for count in erasure_symbols) and
                      all(report['dropouts']['by_side'].values()))
report['passed'] = (lost == 0 and not index_gaps and not status_events and
                    applied_events == len(schedule) and erasures_match and
                    dropouts_on_signal == len(schedule) and
                    report['receiver']['metadata_valid'] == len(rows) and
                    report['receiver']['input_gaps'] == 0 and
                    report['receiver']['decoder_errors'] == 0 and
                    (intact_side is None or
                     report['portaudio']['intact_leg_header_loss'] == 0))
(OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
print(f'Artifacts: {OUT}')
if not report['passed']:
    sys.exit(1)
