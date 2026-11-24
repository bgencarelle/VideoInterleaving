"""Print the frame on screen to the computer's own default printer.

Whatever printer the system has as its default: CUPS (``lpstat -d``, ``lp``)
on Linux and macOS, the default printer on Windows. The frame is saved as a
PNG and handed to the print system, fitted to the page. With no default
printer there is nothing to print to and the request is dropped.

The picture is prepared as for the receipt printer: PRINTER_PAPER 'white'
(default) treats the screen's black background as the white paper, so the
frame prints as a greyscale image on paper; 'black' prints it in colour,
exactly as on screen. PRINTER_ROTATE 'auto' turns a landscape window to run
down the page.
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


def frame_image(rgb, paper='white', rotate='auto'):
    """The frame as a PIL image for paper: greyscale with the screen's
    black as white paper ('white'), or in colour as on screen ('black')."""
    from PIL import Image, ImageOps
    image = Image.fromarray(np.ascontiguousarray(rgb))
    if paper == 'white':
        image = ImageOps.invert(image.convert('L'))
    if rotate == 'auto' and image.width > image.height:
        image = image.rotate(90, expand=True)
    return image


def print_file(path, printer, title='VideoInterleaving frame'):
    """Hand an image file to the print system's ``printer``."""
    if sys.platform.startswith('win'):
        os.startfile(path, 'print')            # the default printer
        return
    subprocess.run(['lp', '-d', printer, '-t', title, '-o', 'fit-to-page', path],
                   check=True, capture_output=True, timeout=30)


def print_frame(rgb, paper='white', rotate='auto', printer=None):
    """Print one frame to ``printer`` (default: the system's). False when
    there is no printer to print to."""
    printer = printer or default_printer()
    if not printer:
        return False
    fd, path = tempfile.mkstemp(prefix='vi_frame_', suffix='.png')
    os.close(fd)
    try:
        frame_image(rgb, paper, rotate).save(path)
        print_file(path, printer)
    finally:
        if not sys.platform.startswith('win'):  # lp has spooled its own copy
            try:
                os.remove(path)
            except OSError:
                pass
    return printer
