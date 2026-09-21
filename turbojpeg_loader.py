from __future__ import annotations

import ctypes.util
import os
from pathlib import Path


def _candidate_paths() -> list[str]:
    candidates: list[str] = []

    for env_var in ("TURBOJPEG_LIB", "TJ_LIBRARY"):
        val = os.environ.get(env_var)
        if val:
            candidates.append(val)

    try:
        import settings

        val = getattr(settings, "TURBOJPEG_LIB", None)
        if val:
            candidates.append(str(val))
    except Exception:
        pass

    found = ctypes.util.find_library("turbojpeg")
    if found:
        candidates.append(found)

    # Common Debian multiarch locations
    common = [
        "/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0",
        "/usr/lib/arm-linux-gnueabihf/libturbojpeg.so.0",
        "/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0",
        "/usr/lib/libturbojpeg.so.0",
        "/usr/local/lib/libturbojpeg.so.0",
    ]
    candidates.extend(common)

    # De-dupe while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def get_turbojpeg():
    """
    Returns a TurboJPEG instance, trying common library locations and respecting overrides.
    Raises RuntimeError with actionable install instructions if unavailable.
    """
    try:
        from turbojpeg import TurboJPEG  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTurboJPEG is not installed. Install with `pip install PyTurboJPEG`."
        ) from e

    last_error: Exception | None = None
    for candidate in _candidate_paths():
        try:
            # If it's a path that exists, pass it explicitly.
            p = Path(candidate)
            if p.exists():
                return TurboJPEG(str(p))
            # Otherwise pass the soname; dlopen will search.
            return TurboJPEG(candidate)
        except Exception as e:
            last_error = e
            continue

    # Try default resolution last
    try:
        return TurboJPEG()
    except Exception as e:
        last_error = e

    hint = (
        "TurboJPEG native library not found.\n"
        "- Debian/Raspbian: `sudo apt install libturbojpeg0`\n"
        "- Or set `TURBOJPEG_LIB=/path/to/libturbojpeg.so.0` (or `TJ_LIBRARY=...`)."
    )
    raise RuntimeError(hint) from last_error

