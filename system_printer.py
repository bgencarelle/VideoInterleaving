"""Print the frame on screen to the computer's own default printer.

Whatever printer the system has as its default: CUPS (``lpstat -d``, ``lp``)
on Linux and macOS, the default printer on Windows. The frame is saved as a
PNG and handed to the print system, fitted to the page. With no default
printer there is nothing to print to and the request is dropped.

The display loop hands over the picture ready for paper: drawn on white
(PRINTER_PAPER 'white') or as on screen, cut to the picture, upright. Here
it is printed as it stands, fitted to the page without cropping; a picture
wider than tall goes on a landscape page so it stays upright and fills it.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def default_printer():
    """The system's default printer, or None."""
    try:
        if sys.platform.startswith('win'):
            out = subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 '(Get-CimInstance Win32_Printer | Where-Object Default).Name'],
                capture_output=True, text=True, timeout=10).stdout.strip()
            return out or None
        if not shutil.which('lpstat'):
            return None
        out = subprocess.run(['lpstat', '-d'], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    # "system default destination: NAME" or "no system default destination"
    # (translated on some systems, but the name always follows the colon).
    if ':' not in out or out.lower().startswith('no '):
        return None
    name = out.split(':', 1)[1].strip()
    return name or None


def frame_image(rgb):
    """The picture as a PIL image, as handed over."""
    from PIL import Image
    return Image.fromarray(np.ascontiguousarray(rgb))


def print_file(path, printer, landscape=False, title='VideoInterleaving frame'):
    """Hand an image file to the print system's ``printer``, fitted to the
    page without cropping, on a landscape page if asked."""
    if sys.platform.startswith('win'):
        os.startfile(path, 'print')            # the default printer
        return
    command = ['lp', '-d', printer, '-t', title, '-o', 'fit-to-page']
    if landscape:
        command += ['-o', 'orientation-requested=4']
    subprocess.run(command + [path], check=True, capture_output=True, timeout=30)


def print_frame(rgb, printer=None):
    """Print one frame to ``printer`` (default: the system's). False when
    there is no printer to print to."""
    printer = printer or default_printer()
    if not printer:
        return False
    fd, path = tempfile.mkstemp(prefix='vi_frame_', suffix='.png')
    os.close(fd)
    try:
        image = frame_image(rgb)
        image.save(path)
        print_file(path, printer, landscape=image.width > image.height)
    finally:
        if not sys.platform.startswith('win'):  # lp has spooled its own copy
            try:
                os.remove(path)
            except OSError:
                pass
    return printer
