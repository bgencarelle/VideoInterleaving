#!/usr/bin/env python3
"""Source-only evaluation of analog sparse DCT residual modes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from tools.v7_sparse_residual import SparseResidualWire
from tools.v7_patch_residual import PatchResidualWire
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_color_metrics import resize_rgb,render as production_render
from tools.v7_color_corpus import face_image,indices
from tools.v7_analog_frame_bench import PROFILES
from tools.v7_local_detail_metrics import unit_image,image_bytes,image_mse,gray_difference,band
from tools.v7_local_detail_source_bench import plane_mse


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--faces',type=int,default=4)
    parser.add_argument('--atoms',type=int,nargs='+',default=[4,8,16])
    parser.add_argument('--partition',choices=('validation','test'),default='test')
    parser.add_argument('--format',choices=('sparse','patch'),default='sparse')
    args=parser.parse_args(argv)
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    rows=[];links=[];size=(192,256)
    write(out/'manifest.json',{'source_only':True,'audio_generated':False,'format':args.format,'atoms':args.atoms,
        'partition':args.partition,'files':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('tools/v7_sparse_residual.py','tools/v7_patch_residual.py','tools/v7_sparse_source_bench.py','tools/v7_geometry_wire.py')}})
    for count in args.atoms:
        parent=SparseResidualWire if args.format=='sparse' else PatchResidualWire
        fields=7 if args.format=='sparse' else 4
        class BudgetWire(parent):ATOMS=count
        for profile in PROFILES:
            baseline=ProductionFrameWire(profile,'3:4');wire=BudgetWire(profile,'3:4')
            for index in indices(args.partition,args.faces):
                source=face_image(index);original=resize_rgb(unit_image(source),size)
                values=wire.values(source);selected=wire.last_diagnostics['geometry_selected']
                coefficients=wire.codec.grid.forward(baseline.values(source))
                projected=wire.clean_projection(coefficients)
                existing=production_render(wire,wire.codec.grid.inverse(projected),size,False)
                forced=wire.with_parameters(coefficients,wire.fitted_modes)
                before=wire.render_received(wire.codec.grid.inverse(wire.clean_projection(forced)),size,False)
                adaptive_coefficients=wire.clean_projection(wire.codec.grid.forward(values))
                adaptive=wire.render_received(wire.codec.grid.inverse(adaptive_coefficients),size,False)
                row={'profile':profile,'atoms':count,'slots':count*fields+3,'source_index':index,'selected':selected,
                     'existing_mse':image_mse(original,existing),'candidate_mse':image_mse(original,before),
                     'fine_bands':{str(f):{'existing':plane_mse(band(gray_difference(original,existing),f)),
                                          'candidate':plane_mse(band(gray_difference(original,before),f))} for f in (24.,48.,72.)}}
                rows.append(row)
                w,h=size;canvas=Image.new('RGB',(4*w,h+40),'white');draw=ImageDraw.Draw(canvas)
                for i,(label,image) in enumerate(zip(('Original','Existing V7 PRE-AUDIO',
                    f'Experimental {args.format} PRE-AUDIO','Experimental adaptive PRE-AUDIO'),(original,existing,before,adaptive))):
                    draw.text((i*w+5,5),label,fill='black');canvas.paste(Image.fromarray(image_bytes(image)),(i*w,40))
                name=f'{profile}-k{count}-face-{index}.png';canvas.save(out/name)
                links.append(f'![{profile} {count} modes {index}]({name})')
    summary={'source_only':True,'audio_generated':False,'summaries':[]}
    for count in args.atoms:
        for profile in PROFILES:
            group=[r for r in rows if r['atoms']==count and r['profile']==profile]
            summary['summaries'].append({'profile':profile,'atoms':count,'slots':count*fields+3,'pictures':len(group),
                'source_selections':sum(r['selected'] for r in group),
                'overall_mse_wins':sum(r['candidate_mse']<r['existing_mse'] for r in group),
                'mean_existing_mse':sum(r['existing_mse'] for r in group)/len(group),
                'mean_candidate_mse':sum(r['candidate_mse'] for r in group)/len(group),
                'fine_band_means':{str(f):{'existing':sum(r['fine_bands'][str(f)]['existing'] for r in group)/len(group),
                    'candidate':sum(r['fine_bands'][str(f)]['candidate'] for r in group)/len(group)} for f in (24.,48.,72.)}})
    write(out/'measurements.json',rows);write(out/'SUMMARY.json',summary)
    (out/'REPORT.md').write_text('# Sparse residual source-only screen\n\nAll reconstructions are BEFORE audio; none are received images.\n\n'+'\n\n'.join(links))


if __name__=='__main__':main()
