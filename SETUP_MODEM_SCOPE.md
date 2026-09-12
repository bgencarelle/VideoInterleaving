# Full application setup, including modem and scope

Run `bash setup_app.sh` from the repository root. No mode arguments are needed.
The installer starts from `main` commit `f840677faf66531fda8ba1809395bf3161932335`;
its full application/service setup is retained. The former `--modem` delegation
is removed. Runtime decoder selection and clock/scheduler code are unchanged.

On macOS setup detects MacPorts first, then Homebrew, including their standard
installation paths when absent from PATH. MacPorts uses Python 3.12 and matching
Tk, NumPy, SciPy, Pillow and OpenCV ports. Homebrew uses Python 3.12 and matching
Tk, with pip packages for scientific/image dependencies as needed. Both install
PortAudio, GLFW and FFmpeg alongside the existing application libraries.
The main `.venv` uses that manager's interpreter and system site packages.
As in the main installer, an invalid environment is recreated; Python provider
mismatches now count as invalid even when version numbers match.

MacPorts installs use sudo. The package manager supplies Python before the
Python-version preflight check, so an absent Python no longer prevents its own
installation. Errors go to stderr and a failed setup reports a nonzero exit.
No package manager is installed automatically: if neither is found, setup
provides the official installation addresses and stops visibly.

Modem requirements remain in `requirements-modem.txt`. Scope requirements are
in `requirements-scope.txt`; OpenCV is reused from native packages when possible,
otherwise setup installs opencv-python in the venv. The final checks import the
modem dependencies plus OpenCV and mss without opening devices or windows.
FFmpeg is included for scope capture on Debian/Arch/macOS; on RHEL-family systems
its availability depends on enabled media repositories and it must be provided
separately for the optional FFmpeg capture source.

Requirements hashes cover both included requirement files, using Python SHA256
instead of Linux-only md5sum. No decoder-only redirection or alternate full-app
script is created. The standalone bundle remains separate.

MacPorts provides [legacy macOS installers](https://www.macports.org/install.php).
Successful manager installation does not guarantee every full-app port can build
on Sierra; any package/build error remains visible. Native installation trials
are still needed on the target Macs.
