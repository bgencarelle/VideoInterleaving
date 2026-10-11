"""Run the integrated adaptive mono/stereo sample over the public live path."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.v7_adaptive_pipeline import sample_sender_settings


def command_for(action, mode, device, source=None, extra=()):
    settings = sample_sender_settings(mode)
    command = [sys.executable, str(ROOT / 'tools/v7_live.py'), action,
               '--device', str(device)]
    if action == 'send':
        if not source:
            raise ValueError('send requires --video-source')
        command.extend([
            '--profile', settings['profile'], '--source', 'video',
            '--video-source', str(source), '--capture-width', settings['capture_width'],
            '--brightness', settings['brightness'], '--gamma', settings['gamma'],
            '--dct-encode', '--dct-kernel', settings['dct_kernel'],
            '--no-luma-adjust', '--dct-sharpen', 'off', '--dct-clarity', '0'])
    elif action == 'receive':
        # Public receiver auto-detects both adaptive identities and joins stereo
        # legs. The normal live viewer is used, not a reconstructed test panel.
        command.extend(['--nested-smooth', '0', '--audio-muted'])
    else:
        raise ValueError('action must be send or receive')
    return command + list(extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('send', 'receive'))
    parser.add_argument('--mode', choices=('mono', 'stereo'), default='mono')
    parser.add_argument('--device', required=True)
    parser.add_argument('--video-source')
    parser.add_argument('--print-command', action='store_true')
    args, extra = parser.parse_known_args()
    try:
        command = command_for(args.action, args.mode, args.device,
                              args.video_source, extra)
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    if not args.print_command:
        os.chdir(ROOT)
        os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
