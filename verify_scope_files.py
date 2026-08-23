"""
verify_scope_files.py -- confirm the scope files are the right files.

Run from the repo root:  python verify_scope_files.py

Checks each file exists, is the module it claims to be (not another one saved
under the wrong name), compiles, and exposes the symbols the others import.
Catches the failure mode where two downloads get crossed, which shows up much
later as a confusing ImportError.
"""
import hashlib
import os
import sys

# (path, marker that must appear near the top, symbols it must define)
EXPECTED = [
    ("scope_out.py",
     "scope_out.py -- send 2D vector graphics",
     ["Scope", "rasterize", "choose_device", "resolve_device", "scrub",
      "list_output_devices", "warn_if_builtin", "FPS", "LEVEL"]),
    ("scope_bake.py",
     "scope_bake.py -- shared vector-library toolkit",
     ["XYLibrary", "merge", "raster_frame", "SweepSource", "content_bbox",
      "order_paths", "fit_epsilon", "subdivide", "path_length", "Q"]),
    ("scope_display.py",
     "scope_display.py -- the scope-mode engine",
     ["run_scope"]),
    ("test_scope_pair.py",
     "test_scope_pair.py -- output one main/float pair",
     ["main", "render_trace", "advance", "TraceSim"]),
    ("scope_lowpass.py",
     "scope_lowpass.py -- put a deliberately aggressive low-pass",
     ["lowpass_circular", "CascadedOnePole", "bench_frame", "describe"]),
    (os.path.join("utilities", "convert_to_xy.py"),
     "XY Baker",
     ["vectorize", "make_thumb", "process_folder", "PROFILES", "THUMB_W"]),
]

CROSS_CHECK = {
    "scope_out.py": "must NOT import from scope_bake or scope_display",
}


def main():
    ok = True
    print(f"{'file':34s} {'status':10s} md5")
    print("-" * 72)
    for path, marker, symbols in EXPECTED:
        if not os.path.exists(path):
            print(f"{path:34s} {'MISSING':10s} -")
            ok = False
            continue
        raw = open(path, "rb").read()
        md5 = hashlib.md5(raw).hexdigest()[:12]
        text = raw.decode("utf-8", "replace")

        if marker not in text[:2000]:
            first = next((l for l in text.splitlines() if l.strip()), "")
            print(f"{path:34s} {'WRONG FILE':10s} {md5}")
            print(f"{'':34s}   expected a module whose header says: {marker!r}")
            print(f"{'':34s}   got: {first[:60]!r}")
            ok = False
            continue

        try:
            code = compile(text, path, "exec")
        except SyntaxError as e:
            print(f"{path:34s} {'SYNTAX':10s} {md5}  line {e.lineno}: {e.msg}")
            ok = False
            continue

        # lone surrogates in constants break .pyc writing on Python 3.13+
        bad = []

        def walk(c):
            for k in c.co_consts:
                if isinstance(k, str) and any(0xD800 <= ord(ch) <= 0xDFFF for ch in k):
                    bad.append(k[:40])
                elif hasattr(k, "co_consts"):
                    walk(k)

        walk(code)
        if bad:
            print(f"{path:34s} {'SURROGATE':10s} {md5}  {bad[:1]}")
            ok = False
            continue

        missing = [s for s in symbols if f"{s}" not in text]
        if missing:
            print(f"{path:34s} {'INCOMPLETE':10s} {md5}  missing: {missing}")
            ok = False
            continue

        print(f"{path:34s} {'ok':10s} {md5}")

    # scope_out must be self-contained; if it imports the others, files crossed
    if os.path.exists("scope_out.py"):
        t = open("scope_out.py", encoding="utf-8", errors="replace").read()
        for bad_imp in ("from scope_bake import", "from scope_display import",
                        "from scope_out import"):
            if bad_imp in t:
                print(f"\n!! scope_out.py contains {bad_imp!r} -- that line "
                      "belongs to another module.\n"
                      "   The files were crossed when saving. Re-download "
                      "scope_out.py.")
                ok = False

    print()
    print("ALL FILES OK" if ok else "PROBLEMS FOUND -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())