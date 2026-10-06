#!/usr/bin/env python3
"""Decode-first clean-wire localized-detail pilot, with paired source truth."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from animation_modem.v7_core import adapt_packet_for_output
from tools.v7_analog_frame import ProductionFrameWire,render
from tools.v7_analog_frame_bench import first_picture,PROFILES
from tools.v7_local_detail import LocalDetailWire
from tools.v7_local_detail_metrics import (
    band,recovery_values,fixture_values,gray_difference,audio_levels,detail_errors,
    image_bytes,unit_image,attenuate,image_mse)
from tools.v7_color_corpus import face_image,indices,PACKET_SECONDS


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def recovery(truth,received):
    available,correlation,gain,unexplained,relative_error,passed=recovery_values(truth,received)
    if not available:
        return {'passed':False,'reason':'no source detail energy'}
    return {'correlation':correlation,'transfer':gain,'unexplained_error':unexplained,
            'relative_error':relative_error,
            'passed':passed}


def fixture(seed,family,frequency,load,contrast,size=(384,512),direction='mixed'):
    """Generator independent of the candidate's Gaussian-atom dictionary."""
    return fixture_values(seed,family,frequency,load,contrast,*size,direction)


def transmit(wire,values,source,index,volume_reference=None):
    original=wire.values
    wire.values=lambda rgb:values
    try:
        start=time.perf_counter()
        audio=adapt_packet_for_output(wire.encode_packet(source,1,index),96000)
        original_rms,original_peak=audio_levels(audio)
        gain=1.
        if volume_reference is not None:
            audio,gain=attenuate(audio,volume_reference['rms'],volume_reference['peak'])
        packet_ms=(time.perf_counter()-start)*1000
    finally:wire.values=original
    receiver=(LocalDetailWire(wire.profile,wire.layout) if isinstance(wire,LocalDetailWire)
              else ProductionFrameWire(wire.profile,wire.layout))
    start=time.perf_counter()
    _,received,_=first_picture(receiver,audio,index)
    decode_ms=(time.perf_counter()-start)*1000
    rms,peak=audio_levels(audio)
    return receiver,received,{'packet_ms':packet_ms,'decode_ms':decode_ms,
        'rms':rms,'peak':peak,
        'volume_gain':gain,'original_rms':original_rms,'original_peak':original_peak,
        'samples':len(audio)}


def model_gate(wire,expected,receiver,received,size):
    if received is None:return {'passed':False,'picture_available':False}
    coefficients=receiver.codec.grid.forward(received)
    recovered=receiver.parameters(coefficients)
    truth=wire.correction(expected,size);actual=receiver.correction(recovered,size)
    coordinate_error,rms,error=detail_errors(expected,recovered,wire.parameter_scale,truth,actual)
    parameter_error=max(coordinate_error.tolist())
    branch=receiver.geometry_selected(coefficients)
    return {'passed':bool(branch and parameter_error<=.003 and error/max(rms,.01)<=.1),
            'picture_available':True,'intended_branch_received':branch,
            'normalized_parameter_error':parameter_error,'absolute_detail_rms_error':error,
            'normalized_coordinate_errors':coordinate_error.tolist(),
            'intended_detail_rms':rms,'relative_detail_rms_error':error/max(rms,1e-16),
            'floor_assisted':rms<.01,'gate_relative_error':error/max(rms,.01)}


def comparison(path,images,rejected=False):
    labels=('Original','Existing V7','Experimental source-aware V7 BEFORE audio (fold projection)',
            'Experimental source-aware V7 AFTER audio (forced)','Experimental source-aware V7 (adaptive)')
    height,width=images[0].shape[:2]
    # Vertical layout keeps long diagnostic labels readable.
    canvas=Image.new('RGB',(width,(height+30)*len(images)),'white');draw=ImageDraw.Draw(canvas)
    for i,(label,image) in enumerate(zip(labels,images)):
        top=i*(height+30);draw.text((8,top+8),label,fill='black')
        if image is None:draw.text((8,top+45),'SOURCE MODEL REJECTED BEFORE AUDIO' if rejected else 'NO PICTURE DECODED',fill='red')
        else:canvas.paste(Image.fromarray(image_bytes(image)),(0,top+30))
    canvas.save(path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=1)
    p.add_argument('--frequencies',type=float,nargs='+',default=[48.])
    p.add_argument('--loads',type=int,nargs='+',default=[4])
    p.add_argument('--directions',nargs='+',choices=('x','y','mixed'),default=['mixed'])
    p.add_argument('--families',nargs='+',choices=('edges','local-texture','random-texture'),default=['local-texture','random-texture'])
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES))
    p.add_argument('--layout',choices=('3:4','4:3','16:9'),default='3:4')
    p.add_argument('--contrast',type=float,default=.15)
    p.add_argument('--faces',type=int,default=0)
    p.add_argument('--seed',type=int,default=1729)
    args=p.parse_args(argv)
    if args.pairs<1 or args.faces<0 or not 0<args.contrast<=.3 or any(f<=0 or f>96 for f in args.frequencies) or any(n<1 for n in args.loads):
        p.error('invalid pair count, face count, contrast or frequency (maximum 96)')
    out=args.out;out.mkdir(parents=True,exist_ok=False)
    size={'3:4':(384,512),'4:3':(512,384),'16:9':(768,432)}[args.layout]
    manifest={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'code_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                  ('tools/v7_local_detail.py','tools/v7_geometry_wire.py','tools/v7_local_detail_metrics.py','tools/v7_local_detail_bench.py')},
              'clean_wire_only':True,'qualification_only_until_gates_pass':True,
              'resolution_gain_claim':False,'seeds_receiver_visible':False}
    write(out/'manifest.json',manifest)
    gates=[];rows=[];pairs=[];links=[]
    for profile in args.profiles:
        wire=LocalDetailWire(profile,args.layout)
        baseline=ProductionFrameWire(profile,args.layout)
        write(out/f'{profile}-codec.json',wire.record())
        # Preflight independent atoms, including overlap and off-center supports.
        atoms=np.zeros((wire.ATOMS,8));atoms[:,6:]=.16
        atoms[0]=[45.25,-8.1,.18,-.07,.17,-.13,.12,.24]
        atoms[1]=[-3.2,52.1,.08,.11,-.2,.21,.24,.12]
        empty=np.zeros(wire.codec.grid.off[-1])
        prepared=wire.codec.grid.inverse(wire.with_parameters(empty,atoms))
        inverse=float(np.max(abs(wire.parameters(wire.codec.grid.forward(prepared))-atoms)))
        source=image_bytes(render(wire,prepared,size,False))
        _,_,reference_levels=transmit(baseline,baseline.values(source),source,300)
        receiver,received,_=transmit(wire,prepared,source,300,reference_levels)
        gate=model_gate(wire,atoms,receiver,received,size)
        gate.update(profile=profile,kind='known-atoms',inverse_error=inverse)
        gate['passed']=gate['passed'] and inverse<=1e-9;gates.append(gate)
        if not gate['passed']:
            before=render(wire,prepared,size,False)
            after=None if received is None else render(receiver,received,size,False)
            comparison(out/f'{profile}-qualification.png',[unit_image(source),None,before,after,None])
            links.append(f'![Qualification failure]({profile}-qualification.png)')
            continue
        # Warm source kernels and dictionary windows outside measured fitting.
        wire.values(source)
        def prepare_source(source,name,index):
            start=time.perf_counter();adaptive=wire.values(source)
            fit_ms=(time.perf_counter()-start)*1000
            selected=wire.last_diagnostics['geometry_selected']
            expected=wire.fitted_atoms.copy()
            forced=wire.codec.grid.inverse(wire.with_parameters(wire.codec.grid.forward(adaptive),expected))
            before=render(wire,wire.codec.grid.inverse(wire.clean_projection(wire.codec.grid.forward(forced))),size,False)
            ref_prepared=baseline.values(source)
            ref_before=render(baseline,baseline.codec.grid.inverse(wire.clean_projection(baseline.codec.grid.forward(ref_prepared))),size,False)
            from tools.v7_color_metrics import resize_rgb
            original=resize_rgb(unit_image(source),size)
            source_mse=image_mse(original,before);reference_mse=image_mse(original,ref_before)
            return {'source':source,'name':name,'index':index,'fit_ms':fit_ms,'selected':selected,
                    'expected':expected,'forced':forced,'adaptive':adaptive,'before':before,
                    'ref_prepared':ref_prepared,'original':original,
                    'source_mse':source_mse,'reference_mse':reference_mse,
                    'useful':bool(selected and source_mse<=reference_mse)}

        def evaluate(prepared,source_allowed):
            source=prepared['source'];name=prepared['name'];index=prepared['index']
            existing_receiver,existing_values,refstats=transmit(baseline,prepared['ref_prepared'],source,index)
            stats=None;gate=None;after=final=None
            if source_allowed:
                recv,values,stats=transmit(wire,prepared['forced'],source,index,refstats)
                gate=model_gate(wire,prepared['expected'],recv,values,size)
                gate.update(profile=profile,kind=name);gates.append(gate)
                adapt_receiver,adapt_values,_=transmit(wire,prepared['adaptive'],source,index,refstats)
                after=None if values is None else render(recv,values,size,False)
                final=None if adapt_values is None else render(adapt_receiver,adapt_values,size,False)
            existing=None if existing_values is None else render(existing_receiver,existing_values,size,False)
            filename=f'{profile}-{name}.png'
            comparison(out/filename,[prepared['original'],existing,prepared['before'],after,final],not source_allowed)
            links.append(f'![{profile} {name}]({filename})')
            encode_ms=prepared['fit_ms']+(stats['packet_ms'] if stats else 0)
            rows.append({'profile':profile,'source':name,'source_hash':hashlib.sha256(source.tobytes()).hexdigest(),
                         'selected':prepared['selected'],'fit_ms':prepared['fit_ms'],'full_encode_ms':encode_ms,
                         'deadline_missed':encode_ms>PACKET_SECONDS*1000,'source_model_passed':source_allowed,
                         'source_model_mse':prepared['source_mse'],'existing_pre_audio_mse':prepared['reference_mse'],
                         'forced_audio':stats,'existing_audio':refstats,'gate_passed':gate['passed'] if gate else None})
            return prepared['original'],existing,after,final
        for family,direction in ((family,direction) for family in args.families for direction in args.directions):
            for frequency in args.frequencies:
                for load in ([0] if family=='random-texture' else args.loads):
                    for pair in range(args.pairs):
                        prepared=[]
                        for leg in range(2):
                            seed=args.seed+pair*2+leg
                            source=fixture(seed,family,frequency,load,args.contrast,size,direction)
                            prepared.append(prepare_source(source,f'{family}-{direction}-f{frequency}-n{load}-p{pair}-{leg}',1000+pair*2+leg))
                        result={'profile':profile,'family':family,'direction':direction,'frequency':frequency,
                                'load':None if family=='random-texture' else load,'pair':pair}
                        truth=band(gray_difference(prepared[0]['original'],prepared[1]['original']),frequency)
                        source_recovery=recovery(truth,band(gray_difference(prepared[0]['before'],prepared[1]['before']),frequency))
                        source_allowed=source_recovery['passed'] and all(p['useful'] for p in prepared)
                        result.update(source_model=source_recovery,source_model_passed=source_allowed)
                        # Bad source fits remain visible, but do not enter the
                        # experimental audio pipeline or capacity score.
                        outputs=[evaluate(p,source_allowed) for p in prepared]
                        for label,column in (('existing',1),('forced',2),('adaptive',3)):
                            if outputs[0][column] is None or outputs[1][column] is None:
                                result[label]={'passed':False,'reason':'picture unavailable'}
                            else:result[label]=recovery(truth,band(gray_difference(outputs[0][column],outputs[1][column]),frequency))
                        pairs.append(result)
        for index in indices('test',args.faces) if args.faces else []:
            prepared=prepare_source(face_image(index),f'face-{index}',index)
            evaluate(prepared,prepared['useful'])
    qualified=bool(gates) and all(g['passed'] for g in gates)
    transmitted=[r for r in rows if r['forced_audio'] is not None]
    volume_matched=bool(transmitted) and all(
        r['forced_audio']['rms']<=r['existing_audio']['rms']*(1+1e-12) and
        r['forced_audio']['peak']<=r['existing_audio']['peak']*(1+1e-12) for r in transmitted)
    for pair in pairs:pair['valid_for_scoring']=qualified and pair['source_model_passed'] and volume_matched
    summary={'qualification_passed':qualified,'model_checks':len(gates),
              'model_checks_passed':sum(g['passed'] for g in gates),'pictures':len(rows),
             'scoring_valid':qualified and volume_matched,'resolution_gain_claim':False,'matched_power_confirmed':volume_matched,
             'source_models_rejected_before_audio':sum(not r['source_model_passed'] for r in rows),
             'source_eligible_pairs':sum(p['source_model_passed'] for p in pairs),
             'decode_parameter_limit':.003,'decode_detail_relative_rms_limit':.1,
             'max_normalized_parameter_error':max((g.get('normalized_parameter_error',0) for g in gates),default=0),
             'max_detail_relative_rms_error':max((g.get('gate_relative_error',0) for g in gates),default=0),
             'profiles':[]}
    for profile in args.profiles:
        group=[r for r in rows if r['profile']==profile]
        sent=[r for r in group if r['forced_audio'] is not None]
        summary['profiles'].append({'profile':profile,'pictures':len(group),
            'adaptive_selections':sum(r['selected'] for r in group),
            'encode_mean_ms':sum(r['full_encode_ms'] for r in group)/len(group) if group else None,
            'deadline_misses':sum(r['deadline_missed'] for r in group),
            'source_rejections_before_audio':len(group)-len(sent),
            'forced_to_existing_rms_mean_ratio':sum(r['forced_audio']['rms']/r['existing_audio']['rms'] for r in sent)/len(sent) if sent else None,
            'forced_to_existing_peak_max_ratio':max((r['forced_audio']['peak']/r['existing_audio']['peak'] for r in sent),default=None),
            'valid_pair_passes':{label:sum(pair[label]['passed'] for pair in pairs if pair['profile']==profile and pair['valid_for_scoring']) if qualified else None
                for label in ('existing','forced','adaptive')}})
    summary['families']=[]
    for profile in args.profiles:
        for family,direction in ((f,d) for f in args.families for d in args.directions):
            group=[r for r in pairs if r['profile']==profile and r['family']==family and r['direction']==direction]
            entry={'profile':profile,'family':family,'direction':direction,'pairs':len(group),
                   'source_eligible_pairs':sum(r['source_model_passed'] for r in group),
                   'valid_for_scoring':qualified and volume_matched and all(r['source_model_passed'] for r in group)}
            for label in ('existing','forced','adaptive'):
                available=[r[label] for r in group if 'correlation' in r[label]]
                entry[label]={'passes':sum(r[label]['passed'] for r in group if r['valid_for_scoring']) if qualified else None,
                    **{key:sum(r[key] for r in available)/len(available) if available else None
                       for key in ('correlation','transfer','relative_error','unexplained_error')}}
            summary['families'].append(entry)
    write(out/'MODEL-DECODE-VALIDATION.json',gates);write(out/'measurements.json',rows)
    write(out/'paired-recovery.json',{'valid_for_scoring':qualified and volume_matched,'pairs':pairs})
    write(out/'SUMMARY.json',summary)
    (out/'REPORT.md').write_text('# Localized detail pilot\n\n'+
        'Original / Existing V7 / Experimental source-aware V7 BEFORE audio / AFTER audio (forced) / adaptive.\n\n'+
         'Before audio is an encoder projection. All other decoder outputs come from actual modem audio. '+
         'A poor source model is rejected before experimental audio encoding; its before-audio image remains visible. '+
         'Experimental waveform volume is reduced to no greater RMS or peak than the reference before decoding. '+
        'Missing pictures are labeled, not replaced with black frames. '+
        ('Decode qualification passed. Pilot results are not a resolution claim.' if qualified else
         '**Decode qualification FAILED. Paired metrics are diagnostic and cannot establish capacity.**')+'\n\n'+'\n\n'.join(links))
    return 0 if qualified else 2


if __name__=='__main__':raise SystemExit(main())
