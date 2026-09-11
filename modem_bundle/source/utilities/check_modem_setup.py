"""Check modem dependencies without a display, device queries or audio streams."""
import argparse
import importlib
import sys


# Import the actual FFT/filter and Tk bridge modules, not just parent packages.
# Importing tkinter does not create a window or require DISPLAY.
CHECKS = (
    ('numpy', 'numpy from requirements-modem.txt'),
    ('scipy.fft', 'scipy from requirements-modem.txt / python3-scipy'),
    ('scipy.signal', 'scipy from requirements-modem.txt / python3-scipy'),
    ('PIL.Image', 'Pillow from requirements-modem.txt'),
    ('tkinter', 'Python Tk bindings: python3-tk (Debian), python3-tkinter (Fedora), '
                 'tk (Arch), or python-tk matching your Homebrew Python'),
    ('PIL.ImageTk', 'Pillow Tk support'),
    ('PIL._imagingtk', 'Pillow Tk bridge: python3-pil.imagetk (Debian) or pip Pillow'),
    ('sounddevice', 'sounddevice from requirements-modem.txt and PortAudio '
                    '(libportaudio2 on Debian; portaudio on Fedora/Arch/Homebrew)'),
)


def check(quiet=False):
    failures = []
    for module, remedy in CHECKS:
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(module)
            if not quiet:
                print(f'{module}: {exc}\n  Install/check {remedy}', file=sys.stderr)
    if failures:
        return 1
    if not quiet:
        print('Modem dependencies available (NumPy, SciPy, Pillow/Tk and sounddevice/PortAudio).')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quiet', action='store_true', help='Exit status only')
    args = parser.parse_args()
    return check(args.quiet)


if __name__ == '__main__':
    sys.exit(main())
