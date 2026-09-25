#!/usr/bin/env python3
"""Run the real tools/v7_live.py sender and receiver back to back, joined by
an in-process loopback audio device, with or without the fold prototype.

It needs no sound card or PortAudio: a stand-in `sounddevice` module paces
the sender's writes and the receiver's 1,024-sample input callbacks in real
time. The sender shows a still frame (or the built-in moving test pattern).
The headless receiver saves every picture it shows, and each one is scored
against the frame.

    python test_modem_v7/live_loopback.py --frame FRAME [--fold 500] [--seconds 8]
    python test_modem_v7/live_loopback.py --frame FRAME --fold 500 --receiver-fold 0

--receiver-fold defaults to --fold; set it differently to see a mismatch.
"""
import argparse
import queue
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

from common import DISPLAY, REPO, reference, ssimulacra2

RATE = 48000
BLOCK = 1024


def fake_sounddevice():
    """A loopback 'sounddevice': OutputStream.write -> InputStream callbacks."""
    wire = queue.Queue()
    listening = threading.Event()

    class OutputStream:
        def __init__(self, channels=2, dtype='float32', device=None, blocksize=0,
                     samplerate=None, **_):
            self.samplerate = float(samplerate or RATE)
            self.latency = .05

        def __enter__(self):
            listening.wait(120)         # the receiver must be up before we send
            return self

        def __exit__(self, *exc):
            return False

        def write(self, audio):
            audio = np.asarray(audio, np.float32)
            wire.put(audio)
            time.sleep(len(audio)/self.samplerate)   # a DAC consumes in real time

    class InputStream:
        def __init__(self, samplerate=RATE, channels=2, dtype='float32', device=None,
                     blocksize=BLOCK, callback=None, **_):
            self.samplerate, self.blocksize, self.callback = float(samplerate), blocksize, callback
            self._stop = threading.Event()

        def start(self):
            threading.Thread(target=self._run, daemon=True).start()
            listening.set()

        def _run(self):
            pending = np.zeros((0, 2), np.float32)
            period = self.blocksize/self.samplerate
            next_time = time.monotonic()
            while not self._stop.is_set():
                while len(pending) < self.blocksize:
                    try:
                        pending = np.concatenate([pending, wire.get_nowait()])
                    except queue.Empty:
                        break
                if len(pending) >= self.blocksize:
                    block, pending = pending[:self.blocksize], pending[self.blocksize:]
                else:
                    block = np.zeros((self.blocksize, 2), np.float32)   # silence
                self.callback(block, self.blocksize, None, None)
                next_time += period
                time.sleep(max(0.0, next_time - time.monotonic()))

        def stop(self):
            self._stop.set()

        def close(self):
            self._stop.set()

    def query_devices(device=None, kind=None):
        return {'name': 'loopback', 'max_input_channels': 2, 'max_output_channels': 2,
                'default_samplerate': float(RATE)}

    module = types.ModuleType('sounddevice')
    module.OutputStream, module.InputStream = OutputStream, InputStream
    module.query_devices = query_devices
    return module, wire


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frame', type=Path, help='still source frame (default: moving test pattern)')
    ap.add_argument('--fold', type=int, default=0, help='sender --experimental-fold')
    ap.add_argument('--receiver-fold', type=int, help='receiver --experimental-fold (default: --fold)')
    ap.add_argument('--encode-filter', default='box')
    ap.add_argument('--seconds', type=float, default=8.0)
    ap.add_argument('--out', type=Path, default=REPO/'tmp'/'test_modem_v7'/'loopback')
    args = ap.parse_args(argv)
    receiver_fold = args.fold if args.receiver_fold is None else args.receiver_fold
    out = args.out/f'send{args.fold}_recv{receiver_fold}'
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('v7_*.png'):
        old.unlink()

    module, _ = fake_sounddevice()
    sys.modules['sounddevice'] = module
    sys.path.insert(0, str(REPO))
    from tools import v7_live

    frame = Image.open(args.frame).convert('RGB') if args.frame else None
    if frame is not None:
        still = np.asarray(frame)
        v7_live._capture = lambda _args: (lambda: still)

    ap_live = v7_live.parser()
    recv_args = ap_live.parse_args(['receive', '--device', 'loopback', '--headless', '--no-log',
                                    '--save-dir', str(out),
                                    '--experimental-fold', str(receiver_fold)])
    send_args = ap_live.parse_args(['send', '--device', 'loopback', '--source', 'test',
                                    '--encode-filter', args.encode_filter, '--brightness', '1.0',
                                    '--seconds', str(args.seconds), '--no-log',
                                    '--experimental-fold', str(args.fold)])
    v7_live._resolve_send_source(send_args)
    threading.Thread(target=v7_live.run_receive, args=(recv_args,), daemon=True).start()
    try:
        v7_live.run_send(send_args)
    except ValueError as exc:            # what `vi.modem-send` reports as "send failed"
        raise SystemExit(f'V7 send failed: {exc}')
    time.sleep(1.5)                      # let the receiver finish the last packets

    pictures = sorted(out.glob('v7_*.png'))
    print(f'sender fold {args.fold}, receiver fold {receiver_fold}: '
          f'{len(pictures)} pictures shown in {args.seconds:g} s '
          f'(wire {v7_live.P.PULSE_FPS:.2f} packets/s)')
    if frame is not None and pictures:
        ref = reference(frame)
        scores = [ssimulacra2(ref, Image.open(p).convert('RGB').resize(
            DISPLAY, Image.Resampling.BICUBIC)) for p in pictures[3:]]
        if scores:
            print(f'SSIMULACRA2 against the frame (after 3 warm-up pictures): '
                  f'mean {np.mean(scores):.1f}, min {np.min(scores):.1f}, n={len(scores)}')
    print(f'pictures: {out}')


if __name__ == '__main__':
    main()
