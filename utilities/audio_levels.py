#!/usr/bin/env python3
"""Stereo level view: is audio actually arriving, on both legs, at sane
levels? Rules out dumb hardware (muted/dead channel, wrong device, silence)
before anyone blames the decoder. Text VU, works headless.

Run from the repo root:

    .venv/bin/python utilities/audio_levels.py --device "BlackHole 2ch" \\
        --channels 1,2 --seconds 10
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem.audio_common import device, pair  # noqa: E402


def bar(v, width=24, floor=-60.0):
    """v in dBFS; floor..0 maps to fill."""
    fill = int(round((max(v, floor)-floor)/(0-floor)*width))
    return '#'*fill + '.'*(width-fill)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--device', type=device,
                    help='Audio input: index or name fragment')
    ap.add_argument('--channels', type=pair, default=(0, 1),
                    help='1-based channel pair to watch (default 1,2)')
    ap.add_argument('--seconds', type=float, default=10.0,
                    help='How long to watch; 0 runs until Ctrl-C')
    ap.add_argument('--block-ms', type=float, default=250.0)
    args = ap.parse_args(argv)
    import sounddevice as sd
    kw = {} if args.device is None else {'device': args.device}
    with sd.InputStream(channels=max(args.channels)+1, dtype='float32',
                        **kw) as stream:
        rate = float(stream.samplerate)
        print(f'listening on {args.device or "default"} '
              f'channels {args.channels} at {rate:g} Hz')
        block = max(256, int(rate*args.block_ms/1000))
        t0, peaks, rms_acc, n = time.perf_counter(), None, None, 0
        try:
            while not args.seconds or time.perf_counter()-t0 < args.seconds:
                data, _ = stream.read(block)
                ch = np.asarray(data[:, list(args.channels)], float)
                peak = 20*np.log10(np.maximum(
                    np.max(np.abs(ch), axis=0), 1e-9))
                rms = 20*np.log10(np.maximum(
                    np.sqrt(np.mean(ch**2, axis=0)), 1e-9))
                peaks = peak if peaks is None else np.maximum(peaks, peak)
                rms_acc = rms if rms_acc is None else rms_acc+rms
                n += 1
                print(f'L {bar(peak[0])} {peak[0]:6.1f} dBFS pk '
                      f'{rms[0]:6.1f} rms | R {bar(peak[1])} '
                      f'{peak[1]:6.1f} dBFS pk {rms[1]:6.1f} rms',
                      flush=True)
        except KeyboardInterrupt:
            pass
    if n:
        print(f'\npeak hold L/R dBFS: {peaks[0]:.1f} / {peaks[1]:.1f}; '
              f'mean RMS: {rms_acc[0]/n:.1f} / {rms_acc[1]/n:.1f}')
        if peaks[1] < -50 and peaks[0] > -20:
            print('RIGHT leg looks dead while LEFT is alive -- '
                  'check cables/device routing before blaming the decoder.')
        elif peaks[0] < -50 and peaks[1] > -20:
            print('LEFT leg looks dead while RIGHT is alive -- '
                  'check cables/device routing before blaming the decoder.')
        elif max(peaks) < -50:
            print('Silence on both legs -- nothing is arriving. The '
                  'visualizer has no packets to show (it never emits black '
                  'for a bad packet, but an empty wire shows nothing).')
        elif min(peaks) > -3:
            print('Levels pinned near full scale -- possible clipping; '
                  'back the source off.')


if __name__ == '__main__':
    main(sys.argv[1:])
