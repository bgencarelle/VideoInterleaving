#!/usr/bin/env python3
import os
import sys
import ctypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    webps: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        for name in filenames:
            if name.lower().endswith(".webp"):
                webps.append(d / name)
    return webps


def main() -> None:
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
        print(f"Error: {root} is not a directory.")
        sys.exit(1)

    # workers
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
            print("Invalid worker count.")
            sys.exit(1)

    log_choice = input(
        "Write bad files to a log file? [y/N]: "
    ).strip().lower()
    log_file_path: Path | None = None
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

    print("\nCollecting .webp files...")
    files = find_webp_files(root)
    total = len(files)

    if total == 0:
        print(f"No .webp files found under {root}")
        sys.exit(0)

    print(f"Found {total} WebP file(s) under {root}")
    print(f"Using libwebp from: {_libwebp._name}")
    print(f"Workers: {workers}")
    print()

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
            except Exception:
                ok = False

            if not ok:
                bad_files.append(path)
                print(f"\nBAD: {path}")

            processed += 1
            # live progress
            print(
                f"\rChecked: {processed}/{total}  |  Invalid: {len(bad_files)}",
                end="",
                flush=True,
            )

    print()  # newline after progress
    print("\n===========================================")
    print("Validation complete")
    print("===========================================")
    print(f"Total WebP files checked : {total}")
    print(f"Invalid / Corrupt (libwebp) : {len(bad_files)}")

    if log_file_path is not None:
        if bad_files:
            with log_file_path.open("w", encoding="utf-8") as f:
                for p in bad_files:
                    f.write(str(p) + "\n")
            print(f"Bad file list saved to: {log_file_path}")
        else:
            print(f"No bad files. Log file is empty: {log_file_path}")


if __name__ == "__main__":
    main()
