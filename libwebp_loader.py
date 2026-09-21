#!/usr/bin/env python3
"""
libwebp_loader.py

Dynamic loader for libwebp (fast path for .webp decoding via ctypes).

Usage from another module (e.g. image_loader.py):

    from libwebp_loader import init_libwebp

    _libwebp = init_libwebp()  # or pass verbose=True
    if _libwebp is None:
        raise RuntimeError("libwebp not loaded.")

    # _libwebp is now ready with WebPGetInfo / WebPDecodeRGBAInto signatures set.
"""

import ctypes
from ctypes.util import find_library


def init_libwebp(verbose: bool = False):
    """
    Try to locate and load libwebp using common system names and paths.

    Returns:
        A ctypes.CDLL instance with WebPGetInfo / WebPDecodeRGBAInto signatures
        configured, or None if the library could not be loaded.
    """
    libwebp = None

    # 1) Try system/loader default using ctypes.util.find_library
    libname = find_library("webp")
    candidate_libs = []

    if libname:
        # On macOS this often returns something like "libwebp.dylib" or a full path.
        candidate_libs.append(libname)

    # 2) Fallback: common bare names
    candidate_libs.extend([
        "libwebp.so",
        "libwebp.so.7",
        "libwebp.so.6",
        "libwebp.dylib",
        "libwebp-7.dll",
    ])

    # 3) Fallback: common Homebrew locations (Apple Silicon / Intel)
    candidate_libs.extend([
        "/opt/homebrew/opt/webp/lib/libwebp.dylib",   # Apple Silicon default
        "/usr/local/opt/webp/lib/libwebp.dylib",      # Intel default
    ])

    last_error = None

    for lib in candidate_libs:
        try:
            libwebp = ctypes.CDLL(lib)
            if verbose:
                print(f"[libwebp_loader] Using libwebp from: {lib}")
            break
        except OSError as e:
            last_error = e
            continue

    if libwebp is None:
        if verbose:
            print(f"[libwebp_loader] Failed to load libwebp. Last error: {last_error}")
        return None

    # Configure function signatures
    libwebp.WebPGetInfo.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    libwebp.WebPGetInfo.restype = ctypes.c_int

    libwebp.WebPDecodeRGBAInto.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    libwebp.WebPDecodeRGBAInto.restype = ctypes.POINTER(ctypes.c_uint8)

    return libwebp


# Optional: eager load at import time if you want a module-level constant.
# Comment this out if you prefer fully lazy loading.
LIBWEBP = init_libwebp(verbose=False)
