"""Score the nested folds as the viewer sees them.

Pictures go through the live sender wires and the live receiver (as
tools/v7_nested_eval.py does) and then through the receiver's display stages
at their defaults: 4x DCT reconstruction, edge reconstruction at 75% and
luma-guided colour (tools/v7_gl_viewer.dct_reconstruct_planes).  The score is
SSIMULACRA2 against the source at the displayed size, clean and under hiss.

Judge tables and decoder changes with this, not with coefficient error: the
display's edge reconstruction rebuilds detail from clean coefficients, so a
mode that sends more but noisier coefficients can lose here while winning on
squared error.

    .venv/bin/python tools/v7_nested_display_eval.py --modes "mono nested" aspect-mono-500 \
        --pictures a.png b.png [--save tmp/shown]
    NESTED_FOLD_TABLES=tmp/try.npz .venv/bin/python tools/v7_nested_display_eval.py ...

Needs the ssimulacra2 and scikit-image packages.  Synthetic channels only.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

with redirect_stdout(io.StringIO()):
    from tools import v7_nested_eval as E, v7_live, v7_kernel_bench as bench
    from tools import v7_gl_viewer as GL, v7_sk_study as study
    from animation_modem import v7, v7_kernels
    import nested_fold as NF
    from common import score

KERNELS = {'aspect-mono-500': 'viewer_solve', 'mono nested': 'viewer_solve',
           'stereo-slices': None, 'stereo nested': 'upscale_precomp'}
CASES = (None, 'hiss-45', 'hiss-35')


def display(values, edge='on', strength=.75, chroma='guided'):
    planes = GL.float_planes(np.asarray(values, float), v7.V7_GRIDS)
    y, cb, cr = GL.dct_reconstruct_planes(planes, '4x', edge=edge, edge_strength=strength, chroma=chroma)
    up = lambda p: np.asarray(Image.fromarray(p).resize((y.shape[1], y.shape[0]), Image.BICUBIC)) if p.shape != y.shape else p
    y = (y+1)*.5; cb = up(cb)*.5-.5/255; cr = up(cr)*.5-.5/255
    rgb = np.stack((y+1.402*cr, y-.344136*cb-.714136*cr, y+1.772*cb), -1)
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255+.5))
def masks_for(rig, mode, wire, code):
    if 'stereo' in mode or mode == 'stereo-slices':
        for c in range(8): wire.model_for(rig.base, wire.layout_for(c))
        layout = wire.layout_for(code)
        return [wire.luma_sent_mask(layout).reshape(v7.V7_GRIDS[0])]+[np.asarray(m).reshape(g) for m, g in zip(wire.chroma_sent_masks(layout), v7.V7_GRIDS[1:])]
    codec = wire._codec(wire._packet_model(rig.base, code))
    return [np.asarray(codec.sent_luma_mask()).reshape(v7.V7_GRIDS[0])]+[np.asarray(m).reshape(g) for m, g in zip(v7_live._chroma_sent_masks(codec), v7.V7_GRIDS[1:])]
def decode(rig, registry, mode, rgb, case=None, n=3):
    wire = rig.senders[mode]; code = v7.aspect_wire_code((rgb.shape[1], rgb.shape[0]))
    masks = masks_for(rig, mode, wire, code)
    kernel = KERNELS[mode]
    sel = None if kernel is None else registry.select(
        kernel, profile='aspect-mono-500' if 'mono' in mode else 'stereo-slices')
    with redirect_stdout(io.StringIO()):
        values = bench.encode(rgb, sel, masks)
        if 'stereo' in mode: audio = wire.encode(rig.base, [values]*n, aspect_codes=[code]*n).astype(np.float64)
        else: audio = np.concatenate([wire.encode_packet(rig.base, values, i+1, aspect_code=code, source_index=i) for i in range(n)]).astype(np.float64)
        audio = E.wide(audio)
        if case is not None:
            audio = E.impair(audio, {c.name: c for c in tuple(E.CASES)+E.EXTRA_CASES}[case], seed=3)
        shown = rig.shown(mode, audio)
    return np.asarray(shown[-1][1], float) if shown else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pictures', nargs='+', required=True)
    parser.add_argument('--modes', nargs='+', default=list(KERNELS), choices=list(KERNELS))
    parser.add_argument('--save', help='folder for the pictures as displayed (clean link)')
    args = parser.parse_args(argv)
    with redirect_stdout(io.StringIO()):
        rig = E.Rig()
    registry = v7_kernels.open_registry()
    NF.HOLD = False                       # one packet at a time: no held-picture average
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    print(f'tables: {NF.TABLES}')
    for mode in args.modes:
        row = []
        for case in CASES:
            scores = []
            for path in args.pictures:
                rgb = study.portrait(path)
                reference = Image.fromarray(rgb).resize(
                    (4*v7.V7_GRIDS[0][1], 4*v7.V7_GRIDS[0][0]), Image.Resampling.LANCZOS)
                values = decode(rig, registry, mode, rgb, case)
                if values is None:
                    scores.append(-200.0)
                    continue
                shown = display(values)
                scores.append(score(reference, shown)['ssimulacra2'])
                if args.save and case is None:
                    shown.save(Path(args.save)/f"{Path(path).stem}_{mode.replace(' ', '_')}.png")
            row.append(float(np.mean(scores)))
        print(f'{mode:16s} ' + '  '.join(
            f"{case or 'clean'} {value:7.2f}" for case, value in zip(CASES, row)), flush=True)


if __name__ == '__main__':
    main()
