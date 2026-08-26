"""
test_scope_web.py -- does the browser render the same picture as the server?

    python test_scope_web.py                 # needs node on PATH
    python test_scope_web.py --write out.png # also save a side-by-side

templates/scope.html contains a JavaScript port of render_luma, because the
server has no sound card and each visitor renders the trace on their own
hardware at their own AudioContext rate.  That makes it a SECOND
implementation of an algorithm we already have -- the exact situation
test_scope_parity.py exists to prevent, except this one cannot share code
because it is in a different language.

So it gets tested instead.  This extracts boxGrid() and buildTrace() from the
page, runs them in node against a real luminance frame, and compares the
result to scope_bake.render_luma.

It has already caught three real bugs:

  * aspect ignored -- sx and sy were both hardcoded to 0.9, so a 128x171
    source rendered nearly square (x reached +-0.874 instead of +-0.674) and
    every face was stretched sideways.
  * no contrast stretch -- the 2nd..98th percentile mapping was missing, so
    the picture was flat and far too bright: 72.8% of pixels lit against
    python's 55.1%.
  * vertex snapping -- the walk took the nearest path vertex instead of
    interpolating within the segment, putting every sample exactly on a cell
    centre, which read as a dot grid rather than scanlines.

None of those are syntax errors.  `node --check` passes on all three.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

PAGE = os.path.join("templates", "scope.html")
TOL_RANGE = 0.06        # allowed drift in the drawn extent
TOL_LIT = 4.0           # allowed drift in percent of lit pixels


def _fake_audio():
    import types
    if "sounddevice" in sys.modules:
        return
    m = types.ModuleType("sounddevice")
    m.query_devices = lambda *a, **k: {"name": "stub", "default_samplerate": 48000}
    m.OutputStream = object
    m.default = types.SimpleNamespace(device=(0, 1))
    sys.modules["sounddevice"] = m


def _extract_fn(src, name):
    """Pull one function out of the page by brace matching."""
    i = src.index("function " + name)
    depth = 0
    j = src.index("{", i)
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
    raise ValueError(f"unterminated function {name}")


def _subject(w=128, h=171):
    """A matted subject with internal features, at thumbnail scale."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    head = np.exp(-(((xx - w * .5) / (w * .30)) ** 2 + ((yy - h * .52) / (h * .30)) ** 2))
    eyes = (np.exp(-(((xx - w * .38) / 4.) ** 2 + ((yy - h * .43) / 3.) ** 2))
            + np.exp(-(((xx - w * .62) / 4.) ** 2 + ((yy - h * .43) / 3.) ** 2)))
    mouth = np.exp(-(((xx - w * .5) / 12.) ** 2 + ((yy - h * .68) / 3.) ** 2))
    lum = np.clip(head - .8 * eyes - .55 * mouth, 0, 1).astype(np.float32)
    lum[lum < .05] = 0.0
    return lum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", help="save a side-by-side PNG here")
    ap.add_argument("--samples", type=int, default=1600,
                    help="48000/30 by default -- a browser's usual rate")
    args = ap.parse_args()

    if not shutil.which("node"):
        print("SKIP: node is not on PATH, cannot run the page's JavaScript")
        return 0
    if not os.path.exists(PAGE):
        print(f"SKIP: {PAGE} not found (run from the repo root)")
        return 0

    _fake_audio()
    from scope_bake import render_luma

    page = "\n".join(re.findall(r"<script>(.*?)</script>",
                                open(PAGE, encoding="utf-8").read(), re.S))
    lum = _subject()
    h, w = lum.shape
    n = args.samples

    tmp = tempfile.mkdtemp()
    (np.clip(lum, 0, 1) * 255).astype(np.uint8).tofile(os.path.join(tmp, "lum.raw"))
    js_path = os.path.join(tmp, "run.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(_extract_fn(page, "boxGrid") + "\n")
        f.write(_extract_fn(page, "buildTrace") + "\n")
        f.write(f"""
const fs=require('fs');
const w={w},h={h},n={n},trim=0.10,gamma=2.2;
const raw=fs.readFileSync({json.dumps(os.path.join(tmp, 'lum.raw'))});
const data=new Uint8ClampedArray(w*h*4);
for(let i=0;i<w*h;i++){{data[i*4]=raw[i];data[i*4+3]=255;}}
const cells=Math.max(64,n);
let cols=Math.max(8,Math.round(Math.sqrt(cells/(h/w))));
let rows=Math.max(6,Math.round(cols*h/w));
cols=Math.min(cols,w); rows=Math.min(rows,h);
const g=boxGrid(data,w,h,rows,cols);
const out=buildTrace(g,rows,cols,n,trim,gamma,w,h);
fs.writeFileSync({json.dumps(os.path.join(tmp, 'trace.f32'))},
                 Buffer.from(new Float32Array(out).buffer));
console.log(JSON.stringify({{rows:rows,cols:cols}}));
""")
    r = subprocess.run(["node", js_path], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAIL: the page's JavaScript threw\n" + r.stderr[-1500:])
        return 1
    grid = json.loads(r.stdout.strip().splitlines()[-1])

    js = np.fromfile(os.path.join(tmp, "trace.f32"), dtype=np.float32).reshape(-1, 2)
    py = render_luma(lum, n, trim=0.10, gamma=2.2)

    ok = True

    def check(name, a, b, tol):
        nonlocal ok
        good = abs(a - b) <= tol
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:22} python {a:+.3f}  js {b:+.3f}"
              f"  (tol {tol})")

    print(f"grid {grid['cols']}x{grid['rows']}, {n} samples\n")
    print("drawn extent -- catches the aspect bug")
    check("x min", float(py[:, 0].min()), float(js[:, 0].min()), TOL_RANGE)
    check("x max", float(py[:, 0].max()), float(js[:, 0].max()), TOL_RANGE)
    check("y min", float(py[:, 1].min()), float(js[:, 1].min()), TOL_RANGE)
    check("y max", float(py[:, 1].max()), float(js[:, 1].max()), TOL_RANGE)

    pa = (py[:, 0].max() - py[:, 0].min()) / max(py[:, 1].max() - py[:, 1].min(), 1e-9)
    ja = (js[:, 0].max() - js[:, 0].min()) / max(js[:, 1].max() - js[:, 1].min(), 1e-9)
    check("width/height ratio", pa, ja, 0.08)

    print("\nrendered picture -- catches the missing contrast stretch")
    try:
        import cv2
        from scope_bake import preview_frame
        A, B = preview_frame(js, size=320), preview_frame(py, size=320)
        la = 100 * (A[:, :, 1] > 30).mean()
        lb = 100 * (B[:, :, 1] > 30).mean()
        check("lit pixels %", lb, la, TOL_LIT)
        if args.write:
            cv2.imwrite(args.write, cv2.cvtColor(np.hstack([B, A]), cv2.COLOR_RGB2BGR))
            print(f"  wrote {args.write} (left python, right js)")
    except ImportError:
        print("  SKIP: cv2 unavailable, cannot render the comparison")

    print()
    print("WEB RENDERER MATCHES" if ok else
          "WEB RENDERER DIVERGED -- the page will not look like the server")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())