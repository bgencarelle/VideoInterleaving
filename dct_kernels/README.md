# DCT downscale kernels

Every `*.py` file in this folder is a kernel, offered by `vi.modem-send-gui`
(and `vi.modem-send --dct-kernel NAME`) the next time it starts. Add a file to
try one; delete it to remove it. Files starting with `_` are ignored. Press
**R** in the GUI to rescan without restarting (a running sender rescans too).
More folders: `--dct-kernel-dir DIR` (repeatable) or `$V7_KERNEL_DIR`; a later
folder wins when two files share a name.

A kernel changes only which picture is handed to the unchanged DCT, fold and
modem; the wire, profiles and receiver are not touched. It works inside Direct
DCT encode and is hidden with Pixel encode.

## The shortest kernel

A resampling kernel in *sent-pixel* units (`x = 1` is one pixel of the
picture the wire can hold). The host turns it into a window over the DCT:

```python
import numpy as np
LABEL = 'Triangle'
SUPPORT = 1.0                        # half-width of the kernel, in sent pixels
def kernel(x):                       # x is an array of positions
    return np.clip(1.0 - np.abs(x), 0.0, None)
```

Add `PARAMS = {'width': (1.0, 0.5, 2.0, 0.05, 'help text')}` and take `width`
as an argument, and it becomes a row in the GUI that can be moved while
sending. The tuple is `(default, low, high, step, help, integer)`, the last
two optional. Every kernel also gets `luma_mix` and `chroma_mix`: 0 turns the
kernel off for that plane, 1 is as written, above 1 pushes further.

## Hooks (use any)

| hook | what it is |
|---|---|
| `kernel(x, **p)` | 1-D spatial kernel; the host works out its frequency response |
| `response(nu, **p)` | 1-D frequency response; `nu` in cycles per sent pixel, 0.5 = Nyquist of the wire. Both axes; `RADIAL = True` applies it to the radius |
| `gain(ctx, **p)` | a full 2-D gain over the coder grid (`ctx.grid`) |
| `post(grid, ctx, **p)` | non-linear refit of the final plane, in [0, 1], after luma adjustment |

A hook may also take `plane` (0 luma, 1 Cb, 2 Cr) to treat planes differently.
Gain at DC is forced to 1 (brightness never moves); with luma adjustment on,
the luma window is applied to the luminance it aims at, so the two agree.

`ctx` offers: `grid`, `sent` (the model's transmitted rectangle), `mask` (the
coefficients the wire really carries) and `extent` (its bounding box, which
includes the fold's guests), `band()` (radius in sent-band units),
`project(grid)` (keep only the coefficients the wire carries), `reduce(array,
'min'|'max'|'mean')` (a pixel-domain array down to the grid) and, in `post`,
`reference` (the plane before the window).

## What is checked

On load each kernel runs once, with its defaults, on small test pictures. A
file that does not import or fails there is listed at launch and skipped; the
rest still load. A kernel that fails on a live frame is bypassed for that
frame and the message is shown in the GUI. Gain arrays are cached per
parameter set, so dragging a slider only rebuilds the array it changes.

## Shipped kernels

| file | what it does |
|---|---|
| `antiring.py` | keeps the sent picture inside the range the source had nearby: halos and mesh move out, detail contrast is kept (the sharp-and-clean option; about +4 ms per 1080p frame) |
| `lanczos.py`, `mitchell.py`, `magic_kernel_sharp.py` | classic and modern resampling kernels as windows |
| `gaussian.py` | no overshoot at all, at the cost of edge width |
| `band_taper.py` | radial raised-cosine over the sent band, optionally out to the guests |

Compare them without any audio with `tools/v7_kernel_bench.py`
(`--guests`, `--sheet out.png --image picture.png`, `--param
antiring.radius=2`).
