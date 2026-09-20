"""V7 prototype bench: tape, tape-style NR, top-quality MP3/ATRAC, and mono.

Encodes the checked-in face fixture with tools/v7_proto.py at V6's measured
stream level, applies each damage model, decodes, and writes results.json and a
contact sheet. The tile frame is chosen from the damage timeline, never from
decoder output. Run from the repository root:

    .venv/bin/python tools/v7_bench.py [out_dir] [comma,separated,case,filter]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v7_proto as P                                                    # noqa: E402
import v7_media as MD                                                   # noqa: E402
from animation_modem import v6                                          # noqa: E402
from animation_modem.imaging import image_values, prepare_image, values_image  # noqa: E402

ROOT = P.ROOT
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'scratch/v7-bench'
OUT.mkdir(parents=True, exist_ok=True)
FIX = ROOT/'modem_tests/fixtures/v6_face_1110.png'
SECONDS, LEAD, V6_RMS = 6.0, 0.25, 0.1521

tape_i = MD.tape_matrix('type-i')
CASES = [('clean', lambda a: np.asarray(a, np.float32), None)]
CASES += [(n,) + MD.lift_case(n, SECONDS) for n in ('hiss-45', 'lift-severe', 'lift-worn',
                                                    'lift-skew', 'mute-left')]
CASES += [('tape:'+n, MD.tape_matrix(n), None) for n in ('lowpass-10k', 'wow-flutter', 'dropouts',
                                                         'type-i', 'worn-deck')]
CASES += [('nr:pumping (repo)', MD.tape_matrix('nr-pumping'), None),
          ('nr:dolbyB rec, no decode', MD.nr_dolby_enc_only, None),
          ('nr:dolbyB mistrack +3dB', MD.nr_dolby_mistrack(3), None),
          ('nr:dolbyB mistrack -3dB', MD.nr_dolby_mistrack(-3), None),
          ('nr:dbx rec, no decode', MD.dbx_like, None)]
CASES += [('mp3 320k CBR', MD.mp3(320), None), ('mp3 LAME V0', MD.mp3(0, vbr=0), None),
          ('atrac1 SP 292k (MD)', MD.atrac('sp'), None), ('atrac3plus (Hi-MD)', MD.atrac('plus'), None)]
CASES += [('mono sum', MD.mono_sum, None),
          ('type-i tape -> mp3 320k', MD.chain(tape_i, MD.mp3(320)), None),
          ('dolbyB rec -> atrac1 SP', MD.chain(MD.nr_dolby_enc_only, MD.atrac('sp')), None),
          ('mp3 320k -> mono sum', MD.chain(MD.mp3(320), MD.mono_sum), None)]
if len(sys.argv) > 2:
    CASES = [c for c in CASES if any(k in c[0] for k in sys.argv[2].split(','))]


def main():
    vals = image_values(prepare_image(Image.open(FIX)), v6.V6_GRIDS)
    model = P.build_model(FIX, V6_RMS/np.sqrt(1+10**(P.CLOCK_REL_DB/10)))
    frames = int(np.ceil(SECONDS*P.RATE/P.FRAME))
    audio = P.encode_stream(model, vals, frames, lead=LEAD)
    body = audio[int(LEAD*P.RATE):-int(.25*P.RATE)]
    model.scale *= V6_RMS/np.sqrt(np.mean(body**2))
    audio = P.encode_stream(model, vals, frames, lead=LEAD)
    truth = model.coder.forward(vals)/model.coder.gains
    floor = float(np.sqrt(np.mean((P.values_from(model, truth)-vals)**2)))
    rows = []
    for name, damage, events in CASES:
        pick = 20
        if events:
            t, dur, *_ = max(events, key=lambda e: e[2] if LEAD < e[0] < SECONDS-.2 else -1)
            pick = int((t+dur/2-LEAD)*P.RATE//P.FRAME) + 1
        t0 = time.perf_counter()
        results, info = P.decode_stream(model, damage(audio))
        byc = {r.counter: r for r in results}
        err = {c: float(np.sqrt(np.mean((P.values_from(model, r.coeffs)-vals)**2))) for c, r in byc.items()}
        shown, held = byc.get(pick), False
        if shown is None:
            prior = [c for c in byc if c < pick]
            shown, held = (byc[max(prior)] if prior else None), True
        e = np.array(list(err.values())) if err else None
        row = {'case': name, 'frames': frames,
               'id_verified': sum(r.status == 'verified' for r in results),
               'pictures': len(results),
               'mean_rmse': round(float(e.mean()), 4) if e is not None else None,
               'p90_rmse': round(float(np.percentile(e, 90)), 4) if e is not None else None,
               'shown_frame': pick, 'shown_rmse': round(err[shown.counter], 4) if shown else None,
               'shown_held': held, 'seconds': round(time.perf_counter()-t0, 1)}
        row['_tile'] = values_image(P.values_from(model, shown.coeffs), v6.V6_GRIDS) if shown else None
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != '_tile'}), flush=True)
    (OUT/'results.json').write_text(json.dumps(
        {'truncation_floor': floor, 'fps': P.RATE/P.FRAME,
         'rows': [{k: v for k, v in r.items() if k != '_tile'} for r in rows]}, indent=1))
    tw, th, pad, lab, cols = 180, 240, 10, 36, 5
    tiles = [{'case': 'source (prepared)', '_tile': values_image(vals, v6.V6_GRIDS)}] + rows
    sheet = Image.new('RGB', (cols*(tw+pad)+pad, ((len(tiles)+cols-1)//cols)*(th+lab+pad)+36), 'white')
    d = ImageDraw.Draw(sheet)
    d.text((pad, 10), f'V7 prototype, {P.RATE/P.FRAME:.2f} fps, tape / NR / MP3 / ATRAC. Picture RMSE vs source '
                      f'(clean floor {floor:.3f}). Tile = fixed frame from damage timeline.', fill='black')
    for i, r in enumerate(tiles):
        x, y = pad + (i % cols)*(tw+pad), 36 + (i//cols)*(th+lab+pad)
        if r['_tile'] is not None:
            sheet.paste(r['_tile'].resize((tw, th), Image.NEAREST), (x, y))
        d.text((x, y+th+2), r['case'][:30], fill='black')
        if 'mean_rmse' in r and r['mean_rmse'] is not None:
            d.text((x, y+th+16), f"mean {r['mean_rmse']:.3f}  id {r['id_verified']}/{r['frames']}", fill='black')
    sheet.save(OUT/'contact_sheet.png')
    print('wrote', OUT/'contact_sheet.png')


if __name__ == '__main__':
    main()
