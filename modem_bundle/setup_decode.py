"""Decoder dependency setup; use --install to install Python requirements."""
import argparse
import os
from pathlib import Path
import shutil
import shlex
import json
import venv
import subprocess
import sys
import webbrowser

ROOT = Path(__file__).resolve().parent
MACPORTS = 'https://www.macports.org/install.php'
WINGET = 'https://learn.microsoft.com/en-us/windows/package-manager/winget/'


def executable(name, *locations):
    found = shutil.which(name)
    if found:
        return found
    return next((p for p in locations if os.path.isfile(p) and os.access(p, os.X_OK)), None)


def manager_help(non_interactive=False):
    """Offer official installation instructions, never execute downloaded code."""
    if sys.platform == 'darwin':
        port = executable('port', '/opt/local/bin/port')
        brew = executable('brew', '/opt/homebrew/bin/brew', '/usr/local/bin/brew')
        if port or brew:
            print('Available package manager: ' + (port or brew))
            print('See DECODE_SETUP.md for matching Python/Tk and PortAudio guidance.')
            return
        name, url = 'MacPorts', MACPORTS
        print('MacPorts offers a Sierra installer. Choose your exact macOS version;')
        print('matching Apple developer tools may be needed to build packages.')
    elif sys.platform == 'win32':
        winget = executable('winget')
        if winget:
            print('Available package manager: ' + winget)
            print('Tk support comes from your Python installation; see DECODE_SETUP.md.')
            return
        name, url = 'WinGet (Microsoft App Installer)', WINGET
        print('WinGet requires Windows 10 version 1809 or later.')
    else:
        print('See DECODE_SETUP.md for your OS native packages.')
        return
    print(f'{name} was not found. Installation instructions: {url}')
    print('A package manager is optional if your decoder dependencies already work.')
    if non_interactive or not sys.stdin.isatty():
        return
    try:
        answer = input('Open the official installation page? [y/N] ')
    except EOFError:
        return
    if answer.strip().lower() in ('y', 'yes'):
        try:
            opened = webbrowser.open(url)
        except Exception as exc:
            print(f'Cannot open browser: {exc}', file=sys.stderr)
            opened = False
        if not opened:
            print(f'Open this address manually: {url}', file=sys.stderr)
        print('Complete installation, open a new terminal, then rerun setup.')


def command_text(command):
    return subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command)


def run(command):
    print('Running: ' + command_text(command), flush=True)
    subprocess.run(command, check=True)


def native_dependencies(args):
    """Only install extensions for the selected interpreter's package manager."""
    if sys.platform != 'darwin':
        print('Windows: repair the selected Python installation with Tcl/Tk enabled.')
        print('Linux: install the native packages in DECODE_SETUP.md for your Python.')
        print('Windows/macOS sounddevice wheels normally include PortAudio.')
        return
    prefix = str(Path(sys.base_prefix).resolve())
    version = f'{sys.version_info.major}{sys.version_info.minor}'
    port = executable('port', '/opt/local/bin/port')
    brew = executable('brew', '/opt/homebrew/bin/brew', '/usr/local/bin/brew')
    if port and prefix.startswith(str(Path(port).parent.parent) + '/'):
        packages = [f'py{version}-numpy', f'py{version}-scipy', f'py{version}-Pillow']
        if not args.headless:
            packages.append(f'py{version}-tkinter')
        if not args.wav:
            packages.append('portaudio')
        command = ['sudo', port, 'install'] + packages
    elif brew and ('/Cellar/' in prefix or '/Homebrew/' in prefix):
        packages = [] if args.headless else [f'python-tk@{sys.version_info.major}.{sys.version_info.minor}']
        if not args.wav:
            packages.append('portaudio')
        if not packages:
            return
        command = [brew, 'install'] + packages
    else:
        raise RuntimeError('Selected Python is not managed by detected MacPorts/Homebrew. '
                           'Select that manager’s Python using MODEM_PYTHON; see DECODE_SETUP.md.')
    print('Native installation: ' + command_text(command))
    if args.non_interactive or not sys.stdin.isatty():
        raise RuntimeError('Native installation requires an interactive confirmation. Run the displayed command manually.')
    if input('Install these native packages? [y/N] ').strip().lower() not in ('y', 'yes'):
        raise RuntimeError('Native installation declined; no native packages installed.')
    run(command)


def environment():
    target = ROOT / '.venv'
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if target.exists():
        if not (target / 'pyvenv.cfg').is_file() or not python.is_file():
            raise RuntimeError(f'Incomplete environment: {target}. Rename it before retrying.')
        result = subprocess.run([str(python), '-c',
            'import json,sys; print(json.dumps([sys.base_prefix,list(sys.version_info[:2])]))'],
            check=True, capture_output=True, text=True)
        prefix, version = json.loads(result.stdout)
        if Path(prefix).resolve() != Path(sys.base_prefix).resolve() or version != list(sys.version_info[:2]):
            raise RuntimeError(f'{target} belongs to another Python. Select its interpreter or rename the environment.')
    else:
        # Native scientific packages from MacPorts must remain visible.
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(target)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true', help='Create bundle .venv and install requirements')
    parser.add_argument('--native', action='store_true', help='Offer matching native dependency installation')
    parser.add_argument('--wav', action='store_true', help='Omit live audio dependencies')
    parser.add_argument('--headless', action='store_true', help='Omit Tk checks')
    parser.add_argument('--package-manager', action='store_true', help='Show package-manager assistance only')
    parser.add_argument('--non-interactive', action='store_true', help='Never prompt or open a browser')
    args = parser.parse_args()
    if (args.install or args.native) and args.package_manager:
        parser.error('--package-manager is a separate action')
    print(f'Python: {sys.executable} ({sys.version.split()[0]})', flush=True)
    if sys.version_info < (3, 11):
        print('Decoder setup requires Python 3.11 or newer.', file=sys.stderr)
        return 1
    if args.package_manager:
        manager_help(args.non_interactive)
        return 0
    try:
        if args.native:
            native_dependencies(args)
        python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if args.install:
            python = environment()
            requirements = ROOT / ('requirements-decode-core.txt' if args.wav else 'requirements.txt')
            run([str(python), '-m', 'pip', 'install', '-r', str(requirements)])
        elif not python.is_file():
            raise RuntimeError('Decoder environment is missing. Run setup with --install.')
        flags = (['--wav'] if args.wav else []) + (['--headless'] if args.headless else [])
        run([str(python), str(ROOT / 'check_decode_setup.py')] + flags)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Decoder setup FAILED: {exc}', file=sys.stderr)
        print('Resolve the error above, then rerun setup. See DECODE_SETUP.md.', file=sys.stderr)
        manager_help(args.non_interactive)
        return 1
    print('Decoder dependencies ready; audio and display were not opened.')
    receiver = [str(python), str(ROOT / 'modem_receive.py')]
    receiver += ['--wav', 'YOUR_RECORDING.wav'] if args.wav else ['--device', 'YOUR_DEVICE', '--channels', 'YOUR_CHANNELS']
    if args.headless:
        receiver.append('--headless')
    print('Receive command (replace placeholders with your existing selection):')
    print(command_text(receiver))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print('\nDecoder setup cancelled.', file=sys.stderr)
        sys.exit(130)
