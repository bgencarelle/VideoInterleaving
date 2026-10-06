#!/usr/bin/env python3
"""Source-only comparison: original, production projection, old and refined fits.

No modem audio is generated or decoded. A better optimizer must first produce a
better intended picture; successful source metrics do not certify transmission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from numba import njit
from PIL import Image,ImageDraw
from tools.v7_local_detail import LocalDetailWire,source_units,residual_values
from tools.v7_local_detail_fit import refine_local
from tools.v7_local_edge_fit import fit_edges,edge_synthesize
from tools.v7_analog_frame import ProductionFrameWire,render
from tools.v7_analog_frame_bench import PROFILES
from tools.v7_color_corpus import face_image,indices
from tools.v7_color_metrics import resize_rgb
from tools.v7_local_detail_metrics import unit_image,image_bytes,image_mse,gray_difference,band


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def picture_metrics(original,picture):
    result={'mse':image_mse(original,picture)}
    for frequency in (24.,48.,72.):
        difference=band(gray_difference(original,picture),frequency)
        result[f'band_{int(frequency)}_mse']=plane_mse(difference)
    return result


@njit(cache=True)
def add_correction(base,correction):
    result=base.copy()
    for y in range(base.shape[0]):
        for x in range(base.shape[1]):
            for c in range(3):result[y,x,c]+=correction[y,x]
    return result


@njit(cache=True)
def plane_mse(image):
    total=0.
    for value in image.ravel():total+=value*value
    return total/image.size


def comparison(path,images):
    width=images[0].shape[1];height=images[0].shape[0]
    labels=('Original','Existing V7\nPRE-AUDIO fold projection',
            'Experimental source-aware V7\nOLD fit BEFORE audio',
            'Experimental source-aware V7\nREFINED fit BEFORE audio',
            'Experimental source-aware V7\nEDGE prototype BEFORE audio (unmapped)')
    output=Image.new('RGB',(len(images)*width,height+42),'white');draw=ImageDraw.Draw(output)
    for column,(label,image) in enumerate(zip(labels,images)):
        draw.text((column*width+8,5),label,fill='black')
        output.paste(Image.fromarray(image_bytes(image)),(column*width,42))
    output.save(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--faces',type=int,default=12)
    parser.add_argument('--partition',choices=('validation','test'),default='test')
    parser.add_argument('--profiles',choices=PROFILES,nargs='+',default=list(PROFILES))
    parser.add_argument('--atoms',type=int,choices=(2,4,8),default=8)
    parser.add_argument('--edges',action='store_true')
    args=parser.parse_args(argv)
    if args.faces<1:parser.error('faces must be positive')
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    write(out/'manifest.json',{'source_only':True,'audio_generated':False,'partition':args.partition,
        'faces':args.faces,'profiles':args.profiles,'slot_budget':args.atoms*32+3,'source_cache':False,
        'atoms':args.atoms,'edge_prototype':args.edges,'edge_wire_mapping_installed':False,
        'code_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('tools/v7_local_detail.py','tools/v7_local_detail_fit.py','tools/v7_local_edge_fit.py','tools/v7_local_detail_source_bench.py')}})
    rows=[];links=[]
    size=(384,512)
    class BudgetWire(LocalDetailWire):
        ATOMS=args.atoms
    for profile in args.profiles:
        wire=BudgetWire(profile,'3:4',refine=False)
        baseline=ProductionFrameWire(profile,'3:4')
        # Warm compilation independently from source measurements.
        warm=np.zeros((32,32));seed=np.zeros((wire.ATOMS,8));seed[:,6:]=.16
        refine_local(warm,seed)
        for index in indices(args.partition,args.faces):
            source=face_image(index);original=resize_rgb(unit_image(source),size)
            start=time.perf_counter();adaptive=wire.values(source)
            old_ms=(time.perf_counter()-start)*1000
            old_atoms=wire.fitted_atoms.copy()
            coefficients=wire.codec.grid.forward(adaptive)
            projected=wire.clean_projection(coefficients)
            # The two fits use precisely the same displaced-coefficient base.
            fit_size=(192,256)
            base=render(baseline,wire.base_values(projected),fit_size,False)
            target=resize_rgb(source_units(source),fit_size)
            residual=residual_values(target,base)
            start=time.perf_counter();new_atoms=refine_local(residual,old_atoms)
            refinement_ms=(time.perf_counter()-start)*1000
            prepared=[]
            for atoms in (old_atoms,new_atoms):
                model=wire.with_parameters(coefficients,atoms)
                prepared.append(render(wire,wire.codec.grid.inverse(wire.clean_projection(model)),size,False))
            edge_ms=None
            if args.edges:
                zero=np.zeros_like(target)
                source_gray=gray_difference(target,zero)
                start=time.perf_counter();edge_atoms=fit_edges(residual,source_gray,wire.ATOMS)
                edge_ms=(time.perf_counter()-start)*1000
                full_base=render(baseline,wire.base_values(projected),size,False)
                prepared.append(add_correction(full_base,edge_synthesize(edge_atoms,*size)))
            reference_coefficients=baseline.codec.grid.forward(baseline.values(source))
            reference=render(baseline,baseline.codec.grid.inverse(wire.clean_projection(reference_coefficients)),size,False)
            metrics=[picture_metrics(original,picture) for picture in (reference,*prepared)]
            row={'profile':profile,'source_index':index,'source_hash':hashlib.sha256(source.tobytes()).hexdigest(),
                 'old_fit_ms':old_ms,'refinement_ms':refinement_ms,'edge_fit_ms':edge_ms,
                 'existing':metrics[0],'old':metrics[1],'refined':metrics[2],
                 'refined_beats_existing_mse':metrics[2]['mse']<metrics[0]['mse'],
                 'refined_beats_existing_fine_bands':all(metrics[2][f'band_{f}_mse']<metrics[0][f'band_{f}_mse'] for f in (24,48,72))}
            if args.edges:
                row.update(edge=metrics[3],edge_beats_existing_mse=metrics[3]['mse']<metrics[0]['mse'],
                           edge_beats_existing_fine_bands=all(metrics[3][f'band_{f}_mse']<metrics[0][f'band_{f}_mse'] for f in (24,48,72)))
            rows.append(row)
            name=f'{profile}-face-{index}.png'
            comparison(out/name,[original,reference,*prepared]);links.append(f'![{profile} face {index}]({name})')
            # Fixed fractional eye/hair crop. This is a visualization, not a score.
            full=Image.open(out/name);crop=Image.new('RGB',((len(prepared)+2)*192,256+42),'white')
            draw=ImageDraw.Draw(crop)
            crop_labels=('Original','Existing V7 PRE-AUDIO','Old fit PRE-AUDIO','Refined fit PRE-AUDIO','Edge BEFORE audio (unmapped)')
            for column,label in enumerate(crop_labels[:len(prepared)+2]):
                draw.text((column*192+5,5),label,fill='black')
                crop.paste(full.crop((column*384+96,42+32,column*384+288,42+288)),(column*192,42))
            crop_name=f'{profile}-face-{index}-crop.png';crop.save(out/crop_name)
            links.append(f'![Matched eye/hair crop]({crop_name})')
    summary={'source_only':True,'audio_generated':False,'pictures':len(rows),'atom_count':args.atoms,
             'slot_budget':args.atoms*32+3,'edge_wire_mapping_installed':False,'profiles':[]}
    for profile in args.profiles:
        group=[row for row in rows if row['profile']==profile]
        entry={'profile':profile,'pictures':len(group),
               'mse_improvements_over_existing':sum(r['refined_beats_existing_mse'] for r in group),
               'all_fine_band_improvements_over_existing':sum(r['refined_beats_existing_fine_bands'] for r in group),
               'refinement_mean_ms':sum(r['refinement_ms'] for r in group)/len(group)}
        if args.edges:
            entry.update(edge_mse_improvements_over_existing=sum(r['edge_beats_existing_mse'] for r in group),
                         edge_all_fine_band_improvements_over_existing=sum(r['edge_beats_existing_fine_bands'] for r in group),
                         edge_fit_mean_ms=sum(r['edge_fit_ms'] for r in group)/len(group))
        for label in (('existing','old','refined','edge') if args.edges else ('existing','old','refined')):
            entry[label]={key:sum(r[label][key] for r in group)/len(group) for key in group[0][label]}
        summary['profiles'].append(entry)
    write(out/'measurements.json',rows);write(out/'SUMMARY.json',summary)
    (out/'REPORT.md').write_text('# Source-only fitter comparison\n\n'+
        '**No images here were recovered from audio.** Existing V7 is an encoder-side fold projection. '+
        'Both experimental reconstructions are forced intended pictures, including rejected fits. '+
        'The optional edge column is a source-only prototype, using the same displaced-coefficient base; its wire mapping is not installed. '+
        'A source improvement is not a resolution claim or a decode qualification.\n\n'+'\n\n'.join(links))
    return 0


if __name__=='__main__':raise SystemExit(main())
