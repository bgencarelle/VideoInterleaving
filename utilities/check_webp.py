#!/usr/bin/env python3
"""
WebP Validator

Validates WebP files using the same libwebp library path as the main application.
Useful for detecting corrupted or invalid WebP files that might cause issues during playback.
"""
import argparse
import logging
import os
import sys
import ctypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np  # you already depend on this


# --- Load libwebp exactly like your project ---------------------------------

_libwebp = None
for lib in ("libwebp.so", "libwebp.dylib", "libwebp-7", "libwebp.dll"):
    try:
        _libwebp = ctypes.CDLL(lib)
        break
    except OSError:
        continue

if _libwebp is None:
    raise RuntimeError(
        "Could not load libwebp (tried libwebp.so/.dylib/libwebp-7/.dll).\n"
        "Make sure libwebp is installed and on your library path."
    )

# Prototype the functions
_libwebp.WebPGetInfo.argtypes = [
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_libwebp.WebPGetInfo.restype = ctypes.c_int

_libwebp.WebPDecodeRGBA.argtypes = [
    ctypes.c_char_p,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_libwebp.WebPDecodeRGBA.restype = ctypes.POINTER(ctypes.c_uint8)

_libwebp.WebPFree.argtypes = [ctypes.c_void_p]


def validate_webp_with_libwebp(path: Path) -> bool:
    """
    Return True if this WebP file can be decoded by libwebp using the
    same pattern as your ImageLoader._read_webp; False otherwise.
    """
    try:
        with path.open("rb") as f:
            data = f.read()
    except OSError:
        return False

    size = len(data)
    if size == 0:
        return False

    buf = ctypes.create_string_buffer(data)

    w = ctypes.c_int()
    h = ctypes.c_int()

    ok = _libwebp.WebPGetInfo(buf, size, ctypes.byref(w), ctypes.byref(h))
    if not ok:
        return False

    ptr = _libwebp.WebPDecodeRGBA(
        buf,
        size,
        ctypes.byref(w),
        ctypes.byref(h),
    )
    if not ptr:
        return False

    try:
        # Wrap it in a NumPy array just like you do (forces libwebp to actually
        # produce pixel data), then copy to ensure we fully touch memory.
        array = np.ctypeslib.as_array(ptr, shape=(h.value, w.value, 4))
        _ = array.copy()
    finally:
        _libwebp.WebPFree(ptr)

    return True


def find_webp_files(root: Path) -> list[Path]:
    """Find all .webp files recursively under root directory."""
    webps: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        for name in filenames:
            if name.lower().endswith(".webp"):
                webps.append(d / name)
    return webps


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate WebP files using libwebp (same library as main application)."
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=None,
        help="Directory to scan for WebP files (default: current directory)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "-l", "--log-file",
        type=str,
        default=None,
        help="Path to log file for bad files (default: no log file)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Use interactive mode (prompts for all options)"
    )
    return parser.parse_args()


def interactive_mode() -> tuple[Path, int, Optional[Path]]:
    """Interactive mode: prompt user for all options."""
    print("===========================================")
    print("   WebP Validator (uses your libwebp path)")
    print("===========================================\n")

    root_input = input(
        "Enter directory to scan [default: current directory]: "
    ).strip()
    if not root_input:
        root = Path(".").resolve()
    else:
        root = Path(root_input).expanduser().resolve()

    if not root.is_dir():
        logging.error(f"{root} is not a directory.")
        sys.exit(1)

    default_workers = os.cpu_count() or 4
    workers_input = input(
        f"How many parallel workers? [default: {default_workers}]: "
    ).strip()
    if not workers_input:
        workers = default_workers
    else:
        try:
            workers = int(workers_input)
            if workers < 1:
                raise ValueError
        except ValueError:
            logging.error("Invalid worker count.")
            sys.exit(1)

    log_choice = input(
        "Write bad files to a log file? [y/N]: "
    ).strip().lower()
    log_file_path: Optional[Path] = None
    if log_choice in ("y", "yes"):
        lf_input = input(
            "Enter log file path [default: bad_webp_files_libwebp.txt]: "
        ).strip()
        if not lf_input:
            log_file_path = Path("bad_webp_files_libwebp.txt").resolve()
        else:
            log_file_path = Path(lf_input).expanduser().resolve()
        # Truncate / create
        log_file_path.write_text("", encoding="utf-8")

    return root, workers, log_file_path


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Determine mode
    if args.interactive or (args.directory is None and args.workers is None and args.log_file is None):
        # Interactive mode
        root, workers, log_file_path = interactive_mode()
    else:
        # Command-line mode
        if args.directory:
            root = Path(args.directory).expanduser().resolve()
        else:
            root = Path(".").resolve()
        
        if not root.is_dir():
            logging.error(f"{root} is not a directory.")
            sys.exit(1)
        
        if args.workers:
            workers = args.workers
            if workers < 1:
                logging.error("Worker count must be >= 1")
                sys.exit(1)
        else:
            workers = os.cpu_count() or 4
        
        if args.log_file:
            log_file_path = Path(args.log_file).expanduser().resolve()
            # Truncate / create
            log_file_path.write_text("", encoding="utf-8")
        else:
            log_file_path = None

    logging.info("Collecting .webp files...")
    files = find_webp_files(root)
    total = len(files)

    if total == 0:
        logging.warning(f"No .webp files found under {root}")
        sys.exit(0)

    logging.info(f"Found {total} WebP file(s) under {root}")
    logging.info(f"Using libwebp from: {_libwebp._name}")
    logging.info(f"Workers: {workers}")

    bad_files: list[Path] = []
    processed = 0

    # Threaded validation
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(validate_webp_with_libwebp, p): p for p in files
        }

        for fut in as_completed(future_map):
            path = future_map[fut]
            ok = False
            try:
                ok = fut.result()
            except Exception as e:
                logging.debug(f"Exception validating {path}: {e}")
                ok = False

            if not ok:
                bad_files.append(path)
                logging.warning(f"BAD: {path}")

            processed += 1
            # Live progress
            if processed % max(1, total // 100) == 0 or processed == total:
                logging.info(f"Checked: {processed}/{total}  |  Invalid: {len(bad_files)}")

    logging.info("===========================================")
    logging.info("Validation complete")
    logging.info("===========================================")
    logging.info(f"Total WebP files checked: {total}")
    logging.info(f"Invalid / Corrupt (libwebp): {len(bad_files)}")

    if log_file_path is not None:
        if bad_files:
            with log_file_path.open("w", encoding="utf-8") as f:
                for p in bad_files:
                    f.write(str(p) + "\n")
            logging.info(f"Bad file list saved to: {log_file_path}")
        else:
            logging.info(f"No bad files. Log file is empty: {log_file_path}")


if __name__ == "__main__":
    main()
