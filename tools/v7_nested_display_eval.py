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

    .venv/bin/python tools/v7_nested_display_eval.py --pictures a.png b.png \
        [--cases torture] [--save tmp/shown] [--out tmp/table.json]
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

# The modes compared: each aspect profile and the nested fold under test
# against it.  (stereo-slices, deprecated, is the wire stereo nested rides on.)
KERNELS = {'aspect-mono-500': 'viewer_solve', 'mono nested': 'viewer_solve',
           'aspect-fold-500': 'viewer_solve', 'stereo nested': 'upscale_precomp',
           'stereo-slices': None}
KERNEL_PROFILES = {'aspect-mono-500': 'aspect-mono-500', 'mono nested': 'aspect-mono-500',
                   'aspect-fold-500': 'aspect-fold-500', 'stereo nested': 'stereo-slices',
                   'stereo-slices': 'stereo-slices'}
DEFAULT_MODES = ('aspect-mono-500', 'mono nested', 'aspect-fold-500', 'stereo nested')
ALL_CASES = {case.name: case for case in tuple(E.CASES)+E.EXTRA_CASES}
QUICK_CASES = ('clean-96k', 'hiss-45', 'hiss-35')


def display(values, edge='on', strength=.75, chroma='guided', luma_room=None, scale='4x'):
    """The picture as the receiver's display stages show it at their defaults."""
    planes = GL.float_planes(np.asarray(values, float), v7.V7_GRIDS)
    y, cb, cr = GL.dct_reconstruct_planes(planes, scale, edge=edge, edge_strength=strength, chroma=chroma,
                                          luma_room=luma_room)
    up = lambda p: np.asarray(Image.fromarray(p).resize((y.shape[1], y.shape[0]), Image.BICUBIC)) if p.shape != y.shape else p
    y = (y+1)*.5; cb = up(cb)*.5-.5/255; cr = up(cr)*.5-.5/255
    rgb = np.stack((y+1.402*cr, y-.344136*cb-.714136*cr, y+1.772*cb), -1)
    return Image.fromarray(np.uint8(np.clip(rgb, 0, 1)*255+.5))
def last_room(rig, mode):
    """The decoder's account of the last picture's luma coefficients, as the
    live receive loop hands it to the display."""
    if 'stereo' in mode:
        return getattr(rig.dispatcher.slice_wire, 'last_room', None)
    for codec in getattr(rig.dispatcher.aspect_mono_wire, 'nested_codecs', lambda: ())():
        if codec.last_nested:
            return codec.last_room
    return None


def _planes(masks):
    return [np.asarray(mask).reshape(grid) for mask, grid in zip(masks, v7.V7_GRIDS)]


def masks_for(rig, mode, wire, code):
    """Per plane, the coefficients this mode's wire carries for a picture."""
    if mode in ('stereo-slices', 'stereo nested'):
        for other in range(8):
            wire.model_for(rig.base, wire.layout_for(other))
        layout = wire.layout_for(code)
        return _planes([wire.luma_sent_mask(layout), *wire.chroma_sent_masks(layout)])
    if mode == 'aspect-fold-500':
        codec = wire.codec(wire.model_for(rig.base, wire.layout_for(code)))
    else:
        codec = wire._codec(wire._packet_model(rig.base, code))
    return _planes([codec.sent_luma_mask(), *v7_live._chroma_sent_masks(codec)])


def send(rig, mode, wire, values, code, packets):
    """48 kHz stream of ``packets`` packets of one picture, as the live
    sender builds them (see tools/v7_nested_eval.Rig.audio)."""
    if mode in ('stereo-slices', 'stereo nested'):
        return wire.encode(rig.base, [values]*packets, aspect_codes=[code]*packets)
    if mode == 'aspect-fold-500':
        import tone_code
        model, coefficients = wire.encode_coefficients(rig.base, values, code)
        return np.concatenate([
            tone_code.add_tone_code(
                v7.encode_pulse_frame_coeffs(
                    model, coefficients, counter, aspect_code=code,
                    source_index=counter-1, pilot_tones=False, eof_marker=True,
                    pulse_profile_code=wire.pulse_profile_code),
                counter, tone_code.encode_status(wire.status_mode))
            for counter in range(1, packets+1)])
    return np.concatenate([
        wire.encode_packet(rig.base, values, index+1, aspect_code=code, source_index=index)
        for index in range(packets)])


def decode(rig, registry, mode, rgb, case=None, packets=3):
    """(decoded grid values, the decoder's room for the display) of the last
    picture shown, or (None, None) when nothing was."""
    wire = rig.senders[mode]
    code = v7.aspect_wire_code((rgb.shape[1], rgb.shape[0]))
    masks = masks_for(rig, mode, wire, code)
    kernel = KERNELS[mode]
    selection = None if kernel is None else registry.select(kernel, profile=KERNEL_PROFILES[mode])
    with redirect_stdout(io.StringIO()):
        values = bench.encode(rgb, selection, masks)
        audio = E.wide(np.asarray(send(rig, mode, wire, values, code, packets), np.float64))
        if case is not None:
            audio = E.impair(audio, ALL_CASES[case], seed=3)
        shown = rig.shown(mode, audio)
    if not shown:
        return None, None
    return np.asarray(shown[-1][1], float), last_room(rig, mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pictures', nargs='+', required=True)
    parser.add_argument('--modes', nargs='+', default=list(DEFAULT_MODES), choices=list(KERNELS))
    parser.add_argument('--cases', nargs='+', default=list(QUICK_CASES),
                        help="line conditions by name, or 'torture' for the whole "
                             "torture matrix (default: clean and two hiss levels)")
    parser.add_argument('--save', help='folder for the pictures as displayed')
    parser.add_argument('--out', help='write the table here as JSON as it fills')
    parser.add_argument('--kernel', help="downscale kernel for every mode instead of its default "
                        "('none' for the reference)")
    args = parser.parse_args(argv)
    cases = list(ALL_CASES) if args.cases == ['torture'] else args.cases
    for case in cases:
        if case not in ALL_CASES:
            parser.error(f'unknown case {case!r}')
    if args.kernel:
        for mode in KERNELS:
            KERNELS[mode] = None if args.kernel == 'none' else args.kernel
    with redirect_stdout(io.StringIO()):
        rig = E.Rig()
    registry = v7_kernels.open_registry()
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    print(f'tables: {NF.TABLES}  dither {NF.DITHER}  held average {NF.HOLD}  '
          f'smoothing passes {NF.SMOOTH_PASSES}', flush=True)
    pictures = [(Path(path).stem, study.portrait(path)) for path in args.pictures]
    size = (4*v7.V7_GRIDS[0][1], 4*v7.V7_GRIDS[0][0])
    references = [Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS)
                  for _, rgb in pictures]
    table = {}
    for case in cases:
        row = {}
        for mode in args.modes:
            scores, shown_count = [], 0
            for (name, rgb), reference in zip(pictures, references):
                values, room = decode(rig, registry, mode, rgb, None if case == 'clean-96k' else case)
                if values is None:
                    continue
                shown_count += 1
                shown = display(values, luma_room=room)
                scores.append(score(reference, shown)['ssimulacra2'])
                if args.save:
                    shown.save(Path(args.save)/f"{name}_{case}_{mode.replace(' ', '_')}.png")
            row[mode] = {'score': float(np.mean(scores)) if scores else None,
                         'shown': shown_count, 'sent': len(pictures)}
        table[case] = row
        print(f'{case:18s} ' + '  '.join(
            f"{mode} {'none shown' if row[mode]['score'] is None else format(row[mode]['score'], '7.2f')}"
            f" ({row[mode]['shown']}/{row[mode]['sent']})" for mode in args.modes), flush=True)
        if args.out:
            import json
            Path(args.out).write_text(json.dumps(table, indent=1)+'\n')


if __name__ == '__main__':
    main()
