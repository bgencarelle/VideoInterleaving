#!/usr/bin/env python3
"""Dense frequency/phase resolution confirmation through fresh actual V7 packets."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_analog_frame import ProductionFrameWire,render
from tools.v7_fold_surface import FoldSurfaceWire
from tools.v7_analog_frame_bench import first_picture,PROFILES
from tools.v7_color_corpus import face_image,indices,digest_file
from tools.v7_color_resolution_bench import json_write,code_identity
from tools.v7_color_gain_loss_chart import font


def target(size,axis,frequency,phase,contrast):
    w,h=size
    length=w if axis=='x' else h
    coordinate=(np.arange(length)+.5)/length
    wave=.5+contrast/2*np.cos(2*np.pi*frequency*coordinate+phase)
    plane=np.broadcast_to(wave[None,:] if axis=='x' else wave[:,None],(h,w))
    return np.repeat(np.uint8(np.rint(255*plane))[...,None],3,axis=2)


def measure(source,received,axis,frequency):
    weights=np.array([.2126,.7152,.0722])
    a,b=source@weights,received@weights
    n=a.shape[1] if axis=='x' else a.shape[0]
    angle=2*np.pi*frequency*(np.arange(n)+.5)/n
    design=np.stack((np.ones(n),np.cos(angle),np.sin(angle)),-1)
    collapse=0 if axis=='x' else 1
    fa=np.linalg.lstsq(design,a.mean(collapse),rcond=None)[0]
    fb=np.linalg.lstsq(design,b.mean(collapse),rcond=None)[0]
    contrast=float(np.linalg.norm(fb[1:])/max(np.linalg.norm(fa[1:]),1e-12))
    phase=float(np.arctan2(fa[1]*fb[2]-fa[2]*fb[1],fa[1:]@fb[1:]))
    fitted=design@fb
    residual=b-(fitted[None,:] if axis=='x' else fitted[:,None])
    error=float(np.sqrt(np.mean(residual*residual))/max(np.std(a),1e-12))
    passed=contrast>=.2 and abs(phase)<=np.pi/8 and error<=.25
    return {'contrast':contrast,'phase':phase,'off_pattern_residual':error,'passed':bool(passed)}


def summarize(out,rows,configs,frequencies):
    panels=[]
    for profile,layout,axis,contrast,display in sorted({(r['profile'],r['layout'],r['axis'],r['source_contrast'],r['display']) for r in rows}):
        panel={'profile':profile,'layout':layout,'axis':axis,'source_contrast':contrast,'display':display,'curves':{},'limits':{},'gains':{}}
        for config in configs:
            curve=[]
            for frequency in frequencies:
                group=[r for r in rows if (r['profile'],r['layout'],r['axis'],r['source_contrast'],r['display'],r['config'],r['frequency'])==(profile,layout,axis,contrast,display,config,frequency)]
                fraction=sum(r['passed'] for r in group)/len(group) if group else 0.
                curve.append({'frequency':frequency,'passing_fraction':fraction,'phases':len(group)})
            panel['curves'][config]=curve
            # Require a continuous passing band from the starting frequency;
            # isolated high-frequency passes cannot establish the cutoff.
            limit=None
            for point in curve:
                if point['passing_fraction']<.75:
                    break
                limit=point['frequency']
            panel['limits'][config]=limit
        base=panel['limits']['default']
        for config in configs:
            limit=panel['limits'][config]
            panel['gains'][config]=limit/base if base and limit else None
        panels.append(panel)
    json_write(out/'DENSE-OUTCOME.json',{'packets':len(rows)//2,'scored_views':len(rows),
        'unavailable_views':sum(not r['available'] for r in rows),'fidelity_gate_passed':True,
        'definition':'cycles per picture; >=75% of phases pass at every sampled frequency up to cutoff',
        'panels':panels})
    for profile in PROFILES:
        visible=[p for p in panels if p['profile']==profile and p['display']]
        im=Image.new('RGB',(1100,125+36*len(visible)),'#f6f7f8');d=ImageDraw.Draw(im)
        d.text((16,12),profile+' — reliable clean-wire frequency limits',font=font(20,True),fill='#17212b')
        d.text((16,44),'Four phases; 75% passing; contiguous band. x = vertical bars; y = horizontal bars.',font=font(14),fill='#53606c')
        for j,config in enumerate(configs):d.text((330+j*245,80),config,font=font(13,True),fill='#17212b')
        for i,panel in enumerate(visible):
            y=105+36*i;d.text((16,y+7),f"{panel['layout']} / {panel['axis']} / contrast {panel['source_contrast']}",font=font(13),fill='#17212b')
            for j,config in enumerate(configs):
                limit=panel['limits'][config];gain=panel['gains'][config]
                x=330+245*j;d.rectangle((x,y,x+240,y+32),fill='#d6eedf' if gain and gain>=1.5 else 'white')
                text='no established cutoff' if limit is None else f'{limit:g} cycles'+('' if gain is None else f' / {gain:.2f}x')
                d.text((x+6,y+7),text,font=font(13),fill='#17212b')
        im.save(out/f'DENSE-{profile}.png')
    (out/'REPORT.md').write_text('# Dense clean-wire resolution confirmation\n\n'+
        'Each packet is independently acquired and decoded. Continuous frequency limits require at least three of four phases to pass. '
        'Both source contrast levels and x/y axes are reported. No numerical packing ratio is counted as a resolution gain.\n\n'+
        '\n'.join(f'![{p.stem}]({p.name})' for p in sorted(out.glob('DENSE-*.png')))+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--frozen-from',type=Path,required=True)
    parser.add_argument('--layouts',nargs='+',default=['3:4','4:3','16:9'])
    parser.add_argument('--frequencies',nargs='+',type=float,default=[8,12,16,18,20,22,24,26,28,30,32,36,40])
    parser.add_argument('--contrasts',nargs='+',type=float,default=[.05,.3])
    args=parser.parse_args(argv)
    if args.out.exists():
        raise ValueError('use a new output directory for dense confirmation')
    args.out.mkdir(parents=True)
    parent=json.loads((args.frozen_from/'manifest.json').read_text())['args']
    if parent['normalization']!='fold-native':raise ValueError('parent must have validated fold-native mapping')
    for path,digest in json.loads((args.frozen_from/'training.json').read_text()).items():
        if digest_file(ROOT/path)!=digest:raise ValueError('frozen training source changed')
    training=[face_image(i) for i in indices('train',parent['training'])]
    configs=['default','surface-7to3-strips3','surface-3to2-strips3']
    dimensions={'surface-7to3-strips3':(7,3),'surface-3to2-strips3':(3,2)}
    identity=code_identity();identity['dense_files']={str(p.relative_to(ROOT)):digest_file(p) for p in (Path(__file__),ROOT/'tools/v7_fold_surface.py',ROOT/'tools/v7_analog_frame.py')}
    json_write(args.out/'manifest.json',{'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'code':identity,'configs':configs,'phases':[0,np.pi/2,np.pi,3*np.pi/2]})
    rows=[];validation=[]
    for profile in PROFILES:
        for layout in args.layouts:
            width,height=map(int,layout.split(':'))
            size=(round(480*width/height),480)
            with redirect_stdout(io.StringIO()):
                wires={'default':ProductionFrameWire(profile,layout)}
                control=FoldSurfaceWire(profile,layout,1,1,3,training,normalization='fold-native')
                for config,(d,k) in dimensions.items():
                    wires[config]=FoldSurfaceWire(profile,layout,d,k,3,training,normalization='fold-native')
                    previous=args.frozen_from/'models'/f'{config}-{profile}-{layout}.json'
                    if previous.exists():
                        frozen=json.loads(previous.read_text())
                        current=wires[config].record()
                        if any(frozen[field]!=current[field] for field in ('mapping_digest','model_digest')):raise ValueError('frozen mapping changed')
            (args.out/'models').mkdir(exist_ok=True)
            for config,wire in wires.items():json_write(args.out/'models'/f'{config}-{profile}-{layout}.json',wire.record())
            for axis in ('x','y'):
                for contrast in args.contrasts:
                    for frequency in sorted(set(args.frequencies)):
                        for phase_index,phase in enumerate((0,np.pi/2,np.pi,3*np.pi/2)):
                            source=target(size,axis,frequency,phase,contrast)
                            source_index=phase_index
                            reference=adapt_packet_for_output(wires['default'].encode_packet(source,1,source_index),96000)
                            transformed=adapt_packet_for_output(control.encode_packet(source,1,source_index),96000)
                            _,base_values,_=first_picture(wires['default'],reference,source_index)
                            _,control_values,_=first_picture(control,transformed,source_index)
                            error=None if base_values is None or control_values is None else float(np.max(abs(base_values-control_values)))
                            passed=error is not None and error<=1e-9 and np.max(abs(reference-transformed))<=1e-9
                            validation.append(passed)
                            if not passed:raise RuntimeError('dense target transformed fidelity gate failed')
                            for config,wire in wires.items():
                                if config=='default':values=base_values
                                else:
                                    audio=adapt_packet_for_output(wire.encode_packet(source,1,source_index),96000)
                                    _,values,_=first_picture(wire,audio,source_index)
                                for display in (False,True):
                                    row={'profile':profile,'layout':layout,'axis':axis,'frequency':frequency,'phase':phase_index,
                                         'source_contrast':contrast,'config':config,'display':display,'available':values is not None,'passed':False}
                                    if values is not None:row.update(measure(source/255,render(wire,values,size,display),axis,frequency))
                                    rows.append(row)
            json_write(args.out/'measurements.json',rows)
    json_write(args.out/'CONTROL-VALIDATION.json',{'passed':all(validation),'probes':len(validation)})
    summarize(args.out,rows,configs,sorted(set(args.frequencies)))
    print('Computed outcome:',args.out/'DENSE-OUTCOME.json')


if __name__=='__main__':main()
