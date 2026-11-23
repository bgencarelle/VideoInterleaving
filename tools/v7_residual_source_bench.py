#!/usr/bin/env python3
"""Source-first test of fixed high-band residual modes. No audio is generated."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_analog_frame import render as experimental_render
from tools.v7_analog_frame_bench import PROFILES
from tools.v7_color_corpus import face_image,indices
from tools.v7_color_metrics import resize_rgb,render as production_render
from tools.v7_local_detail_metrics import image_mse,gray_difference,band,unit_image,image_bytes
from tools.v7_residual_detail import ResidualDetailWire,project_modes


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def metrics(original,existing,candidate):
    return {'existing_mse':image_mse(original,existing),
            'candidate_mse':image_mse(original,candidate),
            'fine_bands':{str(frequency):{
                'existing':plane_mse(band(gray_difference(original,existing),frequency)),
                'candidate':plane_mse(band(gray_difference(original,candidate),frequency))}
                for frequency in (24.,48.,72.)}}


from numba import njit


@njit(cache=True)
def plane_mse(values):
    total=0.
    for value in values.ravel():total+=value*value
    return total/values.size


def comparison(path,images):
    labels=('Original','Existing V7 PRE-AUDIO fold projection',
            'Experimental fixed residual modes BEFORE audio (forced)',
            'Experimental fixed residual modes BEFORE audio (adaptive)')
    h,w=images[0].shape[:2];output=Image.new('RGB',(w,(h+30)*4),'white');draw=ImageDraw.Draw(output)
    for i,(label,image) in enumerate(zip(labels,images)):
        top=i*(h+30);draw.text((8,top+6),label,fill='black')
        output.paste(Image.fromarray(image_bytes(image)),(0,top+30))
    output.save(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--faces',type=int,default=4)
    parser.add_argument('--partition',choices=('validation','test'),default='test')
    parser.add_argument('--counts',type=int,nargs='+',choices=(64,128,256),default=[64,128,256])
    parser.add_argument('--profiles',choices=PROFILES,nargs='+',default=list(PROFILES))
    args=parser.parse_args(argv)
    if args.faces<1:parser.error('faces must be positive')
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    size=(192,256);rows=[];links=[]
    code_files=('tools/v7_residual_detail.py','tools/v7_residual_source_bench.py')
    write(out/'manifest.json',{'source_only':True,'audio_generated':False,'partition':args.partition,
        'faces':args.faces,'counts':args.counts,'profiles':args.profiles,'fixed_mode_indices':True,
        'entropy_bitstream':False,'digital_quantization':False,
        'code_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in code_files}})
    for profile in args.profiles:
        baseline=ProductionFrameWire(profile,'3:4')
        for count in args.counts:
            wire=ResidualDetailWire(profile,'3:4',count)
            for index in indices(args.partition,args.faces):
                source=face_image(index);original=resize_rgb(unit_image(source),size)
                source_values=baseline.values(source)
                adaptive_values=wire.values(source)
                adaptive_selected=wire.last_diagnostics['geometry_selected']
                coefficients=wire.codec.grid.forward(source_values)
                projected=wire.clean_projection(coefficients)
                existing=production_render(wire,wire.codec.grid.inverse(projected),size,False)
                base=production_render(wire,wire.base_values(projected),size,False)
                modes=wire.fitted_modes.copy()
                encoded=wire.with_parameters(coefficients,modes)
                decoded=wire.parameters(wire.codec.grid.forward(wire.codec.grid.inverse(encoded)))
                detail=wire.correction(decoded,size)
                forced=add_luma(base,detail)
                adaptive=experimental_render(wire,adaptive_values,size,False)
                score=metrics(original,existing,forced)
                saturation=wire.last_diagnostics['saturated_coefficients']
                row={'profile':profile,'count':count,'slots':count+3,'source_index':index,
                    'source_hash':hashlib.sha256(source.tobytes()).hexdigest(),
                    'parameter_inverse_max_error':max_abs_difference(decoded,modes),
                    'saturated_coefficients':saturation,
                    'adaptive_selected':adaptive_selected,
                    'beats_existing_overall_mse':score['candidate_mse']<score['existing_mse'],
                    'beats_existing_all_fine_bands':all(v['candidate']<v['existing'] for v in score['fine_bands'].values()),
                    **score}
                rows.append(row)
                name=f'{profile}-m{count}-face-{index}.png'
                comparison(out/name,[original,existing,forced,adaptive]);links.append(f'![{profile} {count} modes face {index}]({name})')
    summaries=[]
    for profile in args.profiles:
        for count in args.counts:
            group=[r for r in rows if r['profile']==profile and r['count']==count]
            summaries.append({'profile':profile,'count':count,'slots':count+3,'pictures':len(group),
                'overall_mse_wins':sum(r['beats_existing_overall_mse'] for r in group),
                'all_fine_band_wins':sum(r['beats_existing_all_fine_bands'] for r in group),
                'saturated_modes':sum(r['saturated_coefficients'] for r in group),
                'adaptive_selections':sum(r['adaptive_selected'] for r in group),
                'mean_existing_mse':sum(r['existing_mse'] for r in group)/len(group),
                'mean_candidate_mse':sum(r['candidate_mse'] for r in group)/len(group),
                'fine_band_means':{str(f):{'existing':sum(r['fine_bands'][str(f)]['existing'] for r in group)/len(group),
                    'candidate':sum(r['fine_bands'][str(f)]['candidate'] for r in group)/len(group)}
                    for f in (24.,48.,72.)}})
    write(out/'measurements.json',rows);write(out/'SUMMARY.json',{'source_only':True,'audio_generated':False,'summaries':summaries})
    (out/'REPORT.md').write_text('# Fixed residual source-only screen\n\n'+
        '**No audio was generated.** Existing V7 is an encoder-side fold projection. '+
        'The forced residual reconstruction and adaptive source output are both BEFORE audio. '+
        'A source-quality win is required before any modem transmission test.\n\n'+'\n\n'.join(links))


@njit(cache=True)
def max_abs_difference(left,right):
    maximum=0.
    for i in range(len(left)):maximum=max(maximum,abs(left[i]-right[i]))
    return maximum


@njit(cache=True)
def add_luma(base,detail):
    output=base.copy()
    for y in range(base.shape[0]):
        for x in range(base.shape[1]):
            for c in range(3):output[y,x,c]+=detail[y,x]
    return output


if __name__=='__main__':main()
