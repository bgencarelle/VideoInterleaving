#!/usr/bin/env python3
"""Run clean, each-leg, and random-leg live V7 dropout torture cases.

Requires PulseAudio/PipeWire-Pulse plus Xvfb, Xclock, PortAudio, and the V7
live sender/receiver dependencies. Each child saves its raw audio, schedule,
receiver log, screenshot, and strict report under ``tmp/``.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'tools/v7_stereo_dropout_torture.py'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seconds', type=float, default=15.0)
parser.add_argument('--seed', type=int, default=20260929)
parser.add_argument('--receiver-profile', choices=('adaptive', 'fold-500'),
                    default='fold-500')
args = parser.parse_args()
if args.seconds < 3:
    parser.error('--seconds must be at least 3')

out = ROOT / f'tmp/v7-stereo-dropout-suite-{time.strftime("%Y%m%d-%H%M%S")}'
out.mkdir(parents=True, exist_ok=True)
case_reports = {}
for side in ('none', 'left', 'right', 'random'):
    command = [sys.executable, str(CASE), '--side', side,
               '--receiver-profile', args.receiver_profile,
               '--seconds', str(args.seconds), '--seed', str(args.seed)]
    completed = subprocess.run(command, cwd=ROOT, text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    print(completed.stdout, end='', flush=True)
    match = re.search(r'^Artifacts: (.+)$', completed.stdout, re.MULTILINE)
    if not match:
        case_reports[side] = {
            'passed': False,
            'error': f'case did not report artifacts; exit={completed.returncode}',
        }
        continue
    report_path = Path(match.group(1)) / 'report.json'
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        report = {'passed': False, 'error': str(exc)}
    report['process_exit_code'] = completed.returncode
    case_reports[side] = report

passed = all(report.get('passed') and report.get('process_exit_code') == 0
             for report in case_reports.values()) and len(case_reports) == 4
suite_report = {
    'receiver_profile': args.receiver_profile,
    'duration_seconds_per_case': args.seconds,
    'seed': args.seed,
    'cases': case_reports,
    'passed': passed,
}
(out / 'report.json').write_text(json.dumps(suite_report, indent=2) + '\n')
print(json.dumps(suite_report, indent=2))
print(f'Suite artifacts: {out}')
if not passed:
    sys.exit(1)
