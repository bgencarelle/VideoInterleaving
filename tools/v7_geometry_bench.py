#!/usr/bin/env python3
"""Source geometry pilot, production controls and clean real-audio scoring."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_analog_frame import ProductionFrameWire, render
from tools.v7_analog_frame_bench import trial, reports, first_picture, PROFILES
from tools.v7_color_corpus import exposures, digest_file
from tools.v7_color_resolution_bench import json_write
from tools.v7_fold_confirmation import write_outcome
from tools.v7_fold_dense_probe import target, measure, summarize
from tools.v7_geometry_wire import GeometryWire


def gate(baseline,control,source,index):
    a=adapt_packet_for_output(baseline.encode_packet(source,1,index),96000)
    b=adapt_packet_for_output(control.encode_packet(source,1,index),96000)
    _,x,_=first_picture(baseline,a,index)
    _,y,_=first_picture(control,b,index)
    error=None if x is None or y is None else float(np.max(abs(x-y)))
    audio_error=float(np.max(abs(a-b)))
    return {'passed':error is not None and error<=1e-9 and audio_error<=1e-9,
            'decoded_max_abs_error':error,'audio_max_abs_error':audio_error}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--faces',type=int,default=2)
    p.add_argument('--movie-entries',type=int,default=3)
    p.add_argument('--partition',choices=('validation','test'),default='validation')
    p.add_argument('--faces-only',action='store_true',help='Only natural face images, no movie fixtures')
    p.add_argument('--skip-dense',action='store_true',help='Finish after the corpus comparison')
    p.add_argument('--dense-only',action='store_true',help='Only the frequency/phase probes')
    args=p.parse_args(argv)
    out=args.out
    if out.exists():raise ValueError('use a new output directory')
    out.mkdir(parents=True)
    (out/'models').mkdir()
    json_write(out/'manifest.json',{'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'files':{str(f.relative_to(ROOT)):digest_file(f) for f in (Path(__file__),ROOT/'tools/v7_geometry_wire.py',ROOT/'tools/v7_analog_frame.py')},
        'clean_only':True,'frame_independent':True,'parameter_slots':GeometryWire.ATOMS*GeometryWire.PARAMETERS+3})
    assets=[] if args.dense_only else exposures(args.partition,out/'cache',assets=('stills',) if args.faces_only else ('stills','movies'),stills=args.faces,repeats=1)
    json_write(out/'sources.json',[e.record() for e in assets])
    wires={};validation=[];rows=[]
    model_validation=[]
    from tools.v7_geometry_validation import decode_model_gate
    def models(profile,layout):
        key=(profile,layout)
        if key not in wires:
            wires[key]=(ProductionFrameWire(profile,layout),GeometryWire(profile,layout,False),GeometryWire(profile,layout))
            json_write(out/'models'/f'{profile}-{layout}.json',wires[key][2].record())
        return wires[key]
    for profile in PROFILES:
        for exposure in assets:
            entries=[0] if exposure.movie is None else sorted(set(np.rint(np.linspace(0,len(exposure.source_indices)-1,args.movie_entries)).astype(int)))
            baseline,control,candidate=models(profile,exposure.layout)
            for entry in entries:
                source=exposure.image(entry);index=exposure.source_indices[entry]
                check=gate(baseline,control,source,index)
                validation.append(check)
                if not check['passed']:raise RuntimeError('transformed fidelity gate failed')
                model_check=decode_model_gate(profile,exposure.layout,source,index,
                    failure_dir=out/'DECODE-FAILURES'/f'{profile}-{exposure.name}-entry{entry}')
                model_validation.append({'profile':profile,'asset':exposure.name,'entry':int(entry),**model_check})
                json_write(out/'MODEL-DECODE-VALIDATION.json',{'passed':all(r['passed'] for r in model_validation),'checks':model_validation})
                if not model_check['passed']:
                    raise RuntimeError('model decode gate failed; capacity scoring stopped; run v7_geometry_validation.py for diagnostic images')
                for name,wire in (('default',baseline),('geometry',candidate)):
                    row={'profile':profile,'asset':exposure.name+f'-entry{entry}','case':'clean-96k','config':name}
                    # Warm kernels then measure a real new source fit: no cached geometry.
                    wire.values(source);start=time.perf_counter();wire.values(source)
                    row['source_analysis_ms']=(time.perf_counter()-start)*1000
                    row.update(trial(wire,source,index,'clean-96k',2026,out/'trials'/f'{profile}-{name}-{exposure.name}-entry{entry}',480,
                                     3 if exposure.movie is None else 0,False,natural=exposure.movie is None,
                                     resolution_target=exposure.name.startswith('v7_pixel_motion')))
                    if name=='geometry':row.update(wire.last_diagnostics)
                    rows.append(row)
                # Side-by-side evidence, not a selected winner montage.
                folders=[out/'trials'/f'{profile}-{name}-{exposure.name}-entry{entry}' for name in ('default','geometry')]
                images=[Image.open(folders[0]/'source.png'),*(Image.open(f/'display.png') for f in folders)]
                w,h=images[0].size
                comparison=Image.new('RGB',(3*w,h+30),'white');draw=ImageDraw.Draw(comparison)
                for i,(label,image) in enumerate(zip(('Original','Existing V7','Experimental source-aware V7'),images)):
                    comparison.paste(image,(i*w,30));draw.text((i*w+8,8),label,fill='black')
                (out/'comparisons').mkdir(exist_ok=True)
                comparison.save(out/'comparisons'/f'{profile}-{exposure.name}-entry{entry}.png')
    json_write(out/'CONTROL-VALIDATION.json',{'passed':all(r['passed'] for r in validation),'trials':validation})
    json_write(out/'rows.json',rows);reports(out,rows);write_outcome(out,rows)
    json_write(out/'ENCODE-OUTCOME.json',[
        {'profile':profile,'config':config,'trials':len(group),
         'geometry_selected':sum(r.get('geometry_selected',False) for r in group),
         'mean_encode_ms':float(np.mean([r['encode_ms'] for r in group])),
         'max_encode_ms':max(r['encode_ms'] for r in group),
         'encode_deadline_misses':sum(r['encode_deadline_missed'] for r in group),
         'packet_seconds':group[0]['packet_seconds']}
        for profile in PROFILES for config in ('default','geometry')
        if (group:=[r for r in rows if r['profile']==profile and r['config']==config])
    ])
    if args.skip_dense:
        with (out/'REPORT.md').open('a') as report:
            report.write('\nSource / production / geometry comparisons: `comparisons/`. '
                         'Every image is independently fitted and acquired; source fitting is included in encoding cost.\n')
        print('Completed computed reports:',out/'OUTCOME.json')
        return
    # Dense probes use the same fitted source model, no phase/frequency side data.
    dense=[];frequencies=[8,12,16,20,24,28,32,40,48,56,64]
    for profile in PROFILES:
        for layout in ('3:4','4:3','16:9'):
            baseline,control,candidate=models(profile,layout)
            width,height=map(int,layout.split(':'));size=(round(480*width/height),480)
            for axis in ('x','y'):
                for contrast in (.05,.3):
                    for frequency in frequencies:
                        for phase_index,phase in enumerate((0,np.pi/2,np.pi,3*np.pi/2)):
                            source=target(size,axis,frequency,phase,contrast)
                            check=gate(baseline,control,source,phase_index)
                            validation.append(check)
                            if not check['passed']:raise RuntimeError('dense fidelity gate failed')
                            model_check=decode_model_gate(profile,layout,source,phase_index,
                                failure_dir=out/'DECODE-FAILURES'/f'{profile}-{layout}-{axis}-{contrast}-{frequency}-{phase_index}')
                            model_validation.append({'profile':profile,'layout':layout,'axis':axis,'frequency':frequency,
                                'phase_index':phase_index,'source_contrast':contrast,**model_check})
                            if not model_check['passed']:
                                json_write(out/'MODEL-DECODE-VALIDATION.json',{'passed':False,'failed_dense_probe':
                                    {'profile':profile,'layout':layout,'axis':axis,'frequency':frequency,'phase_index':phase_index,'source_contrast':contrast,**model_check}})
                                raise RuntimeError('dense model decode gate failed; capacity scoring stopped')
                            pictures=[Image.fromarray(source)]
                            for name,wire in (('default',baseline),('geometry',candidate)):
                                audio=adapt_packet_for_output(wire.encode_packet(source,1,phase_index),96000)
                                _,values,_=first_picture(wire,audio,phase_index)
                                for display in (False,True):
                                    row={'profile':profile,'layout':layout,'axis':axis,'frequency':frequency,'phase_index':phase_index,
                                         'source_contrast':contrast,'config':name,'display':display,'available':values is not None,'passed':False}
                                    if values is not None:row.update(measure(source/255,render(wire,values,size,display),axis,frequency))
                                    if name=='geometry':row['geometry_selected']=wire.last_diagnostics['geometry_selected']
                                    dense.append(row)
                                    if display and frequency in (20,64) and values is not None:
                                        pictures.append(Image.fromarray(np.uint8(np.clip(render(wire,values,size,True),0,1)*255)))
                            if frequency in (20,64) and len(pictures)==3:
                                folder=out/'dense-comparisons';folder.mkdir(exist_ok=True)
                                w,h=size
                                comparison=Image.new('RGB',(3*w,h+30),'white');draw=ImageDraw.Draw(comparison)
                                for i,(label,image) in enumerate(zip(('Original','Existing V7','Experimental source-aware V7'),pictures)):
                                    comparison.paste(image,(i*w,30));draw.text((i*w+8,8),label,fill='black')
                                comparison.save(folder/f'{profile}-{layout}-{axis}-contrast{contrast}-cycles{frequency}-phase{phase_index}.png')
    json_write(out/'CONTROL-VALIDATION.json',{'passed':all(r['passed'] for r in validation),'trials':validation})
    json_write(out/'MODEL-DECODE-VALIDATION.json',{'passed':all(r['passed'] for r in model_validation),'checks':model_validation})
    dense_out=out/'dense';dense_out.mkdir()
    json_write(dense_out/'measurements.json',dense)
    summarize(dense_out,dense,['default','geometry'],frequencies)
    with (out/'REPORT.md').open('a') as report:
        report.write('\n## Source-aware geometry pilot\n\n83 ordinary luma slots carry eight global Fourier atoms with repeated coarse frequencies, fine frequency offsets, companded amplitudes and three selection markers; remaining picture coefficients are retained. '
                     'The receiver gets no source, fitted state, or frequency label. '
                     'This replaces coefficient detail, so natural-image regressions and computation cost count against the candidate. '
                     'Source fitting is included in encoding timing; no source cache is used.\n\n[Dense frequency/phase charts](dense/REPORT.md)\n\n'
                     'All comparison images: `comparisons/` (source / production / geometry).\n')
    print('Completed computed reports:',out/'OUTCOME.json',dense_out/'DENSE-OUTCOME.json')


if __name__=='__main__':main()
