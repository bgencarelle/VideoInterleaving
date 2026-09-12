"""Verify only selected decoder dependencies; no GUI or audio streams opened."""
import argparse
import importlib
import sys


def required_modules(wav=False, headless=False):
    modules = ['numpy', 'scipy.fft', 'scipy.signal', 'PIL.Image']
    if not wav:
        modules.append('sounddevice')
    if not headless:
        modules.extend(('tkinter', 'PIL.ImageTk', 'PIL._imagingtk'))
    return modules


def check(wav=False, headless=False, quiet=False):
    failures = []
    for name in required_modules(wav, headless):
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append((name, str(exc)))
    if not quiet:
        if failures:
            for name, error in failures:
                print(f'{name}: {error}', file=sys.stderr)
            print('See DECODE_SETUP.md for Python and native dependencies.', file=sys.stderr)
        else:
            print('Selected decoder dependencies are available.')
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', action='store_true', help='WAV input; omit live audio dependencies')
    parser.add_argument('--headless', action='store_true', help='No display; omit Tk dependencies')
    parser.add_argument('--quiet', action='store_true', help='Exit status only')
    args = parser.parse_args()
    return check(args.wav, args.headless, args.quiet)


if __name__ == '__main__':
    sys.exit(main())
