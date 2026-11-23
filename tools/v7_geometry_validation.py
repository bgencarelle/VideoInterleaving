#!/usr/bin/env python3
"""Separate source-model loss, parameter mapping and actual V7 transport loss.

The forced branch is diagnostic: it does not claim to be the adaptive encoder's
selected output, and it is never counted as a resolution gain on natural faces.
"""
import argparse
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
from tools.v7_color_corpus import exposures,digest_file,PACKET_SECONDS
from tools.v7_color_metrics import resize_rgb,raster_size,picture_metrics
from tools.v7_color_resolution_bench import json_write
from tools.v7_geometry_wire import GeometryWire,fit_atom,synthesize


class DiagnosticWire(GeometryWire):
    """Expose a rejected fit without changing the current encoder's code."""
    def fit_residual(self,residual):
        self.fitted_atoms=super().fit_residual(residual)
        return self.fitted_atoms

    def force_values(self,rgb):
        adaptive=super().values(rgb)
        # All untouched coefficients are identical in both branches. Restore
        # only the parameter carriers and marker to the forced geometry values.
        coefficients=self.codec.grid.forward(adaptive)
        return self.codec.grid.inverse(self.with_parameters(coefficients,self.fitted_atoms))


def validate_mapping(wire):
    atoms=np.zeros((wire.ATOMS,4))
    atoms[0]=[13.25,-4.1,.12,-.07]
    atoms[1]=[-3.75,17.1,-.04,.02]
    empty=np.zeros(wire.codec.grid.off[-1])
    values=wire.codec.grid.inverse(wire.with_parameters(empty,atoms))
    recovered=wire.parameters(wire.codec.grid.forward(values))
    inverse_error=float(np.max(abs(atoms-recovered)))
    # Isolate continuous-frequency fitting from image compression and transport.
    atom=atoms[:1]
    known=synthesize(atom,256,192)
    fitted=np.asarray(fit_atom(known,13.,-4.))
    fit_error=float(np.max(abs(fitted-atom[0])))
    return {'parameter_inverse_max_abs_error':inverse_error,
            'known_atom_fit_max_abs_error':fit_error,
            'passed':inverse_error<=1e-9 and fit_error<=.01}


def decode_model_gate(profile,layout,source,index,failure_dir=None):
    """Exercise geometry even when normal source selection would fall back."""
    encoder=DiagnosticWire(profile,layout)
    prepared=encoder.force_values(source)
    expected=encoder.parameters(encoder.codec.grid.forward(prepared))
    encoder.values=encoder.force_values
    audio=adapt_packet_for_output(encoder.encode_packet(source,1,index),96000)
    receiver=GeometryWire(profile,layout)
    _,received,_=first_picture(receiver,audio,index)
    record={'passed':False,'picture_available':received is not None,
            'intended_branch':'forced geometry','normalized_parameter_tolerance':.003,
            'relative_correction_tolerance':.1}
    if received is None:
        if failure_dir is not None:
            folder=Path(failure_dir);folder.mkdir(parents=True,exist_ok=True)
            Image.fromarray(source).save(folder/'original.png')
            json_write(folder/'FAILURE.json',record)
        return record
    coefficients=receiver.codec.grid.forward(received)
    recovered=receiver.parameters(coefficients)
    branch=receiver.geometry_selected(coefficients)
    parameter_error=float(np.max(abs((expected-recovered)/receiver.parameter_scale)))
    size=raster_size(source,240)
    intended_correction=receiver.correction(expected,size)
    received_correction=receiver.correction(recovered,size)
    correction_error=float(np.sqrt(np.mean((intended_correction-received_correction)**2)))
    reference_rms=max(float(np.sqrt(np.mean(intended_correction**2))),.01)
    relative_error=correction_error/reference_rms
    record.update(geometry_branch_received=branch,normalized_parameter_max_abs_error=parameter_error,
                  relative_correction_rms_error=relative_error,
                  passed=branch and parameter_error<=.003 and relative_error<=.1)
    if not record['passed'] and failure_dir is not None:
        folder=Path(failure_dir);folder.mkdir(parents=True,exist_ok=True)
        size=raster_size(source,480)
        projected=encoder.clean_projection(encoder.codec.grid.forward(prepared))
        before=render(encoder,encoder.codec.grid.inverse(projected),size,True)
        after=render(receiver,received,size,True)
        images=(resize_rgb(source/255,size),before,after)
        w,h=size;comparison=Image.new('RGB',(3*w,h+30),'white');draw=ImageDraw.Draw(comparison)
        for i,(label,image) in enumerate(zip(('Original','Intended model BEFORE audio','Recovered model AFTER V7 audio'),images)):
            picture=Image.fromarray(np.uint8(np.clip(image,0,1)*255))
            picture.save(folder/f'{i}.png');comparison.paste(picture,(i*w,30));draw.text((i*w+8,8),label,fill='black')
        comparison.save(folder/'comparison.png');json_write(folder/'FAILURE.json',record)
    return record


def main(argv=None):
    from tools.v7_geometry_bench import gate
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--faces',type=int,default=4)
    p.add_argument('--partition',choices=('validation','test'),default='test')
    args=p.parse_args(argv)
    out=args.out
    if out.exists():raise ValueError('use a new output directory')
    out.mkdir(parents=True)
    json_write(out/'manifest.json',{'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'files':{str(f.relative_to(ROOT)):digest_file(f) for f in (Path(__file__),ROOT/'tools/v7_geometry_wire.py',ROOT/'tools/v7_analog_frame.py')},
        'forced_geometry_is_diagnostic':True,'source_cache':False,'clean_only':True})
    assets=exposures(args.partition,out/'cache',assets=('stills',),stills=args.faces,repeats=1)
    json_write(out/'sources.json',[e.record() for e in assets])
    rows=[];controls=[];mapping=[];model_checks=[]
    for profile in PROFILES:
        wires={}
        for exposure in assets:
            if exposure.layout not in wires:
                wire=DiagnosticWire(profile,exposure.layout)
                test=validate_mapping(wire);mapping.append({'profile':profile,'layout':exposure.layout,**test})
                if not test['passed']:raise RuntimeError('source parameter mapping failed before scoring')
                wires[exposure.layout]=(ProductionFrameWire(profile,exposure.layout),GeometryWire(profile,exposure.layout,False),wire)
            baseline,control,wire=wires[exposure.layout]
            source=exposure.image(0);index=exposure.source_indices[0];size=raster_size(source,480)
            test=gate(baseline,control,source,index);controls.append(test)
            if not test['passed']:raise RuntimeError('transformed transport control failed')
            # Warm compilation/source kernels, then fit the same source anew.
            wire.force_values(source)
            start=time.perf_counter();prepared=wire.force_values(source)
            analysis_ms=(time.perf_counter()-start)*1000
            coefficients=wire.codec.grid.forward(prepared)
            predicted=wire.clean_projection(coefficients)
            before=render(wire,wire.codec.grid.inverse(predicted),size,True)
            before_raw=render(wire,wire.codec.grid.inverse(predicted),size,False)
            expected=wire.parameters(predicted)
            # Force only the diagnostic branch's values; do not pass source
            # atoms to the receiver. Its reconstruction is otherwise unchanged.
            original_values=wire.values
            wire.values=wire.force_values
            try:
                start=time.perf_counter()
                audio=adapt_packet_for_output(wire.encode_packet(source,1,index),96000)
                encode_ms=(time.perf_counter()-start)*1000
            finally:wire.values=original_values
            receiver=GeometryWire(profile,exposure.layout)
            _,received,_=first_picture(receiver,audio,index)
            if received is None:raise RuntimeError('forced diagnostic picture unavailable')
            after=render(receiver,received,size,True)
            after_raw=render(receiver,received,size,False)
            received_coeffs=receiver.codec.grid.forward(received)
            recovered=receiver.parameters(received_coeffs)
            model_checks.append({'profile':profile,'asset':exposure.name,
                                 **decode_model_gate(profile,exposure.layout,source,index)})
            ref_audio=adapt_packet_for_output(baseline.encode_packet(source,1,index),96000)
            _,ref_values,_=first_picture(baseline,ref_audio,index)
            if ref_values is None:raise RuntimeError('production reference picture unavailable')
            reference=render(baseline,ref_values,size,True)
            original=resize_rgb(source/255,size)
            row={'profile':profile,'asset':exposure.name,'adaptive_would_select_geometry':wire.last_diagnostics['geometry_selected'],
                 'diagnostic_geometry_selected_by_receiver':wire.geometry_selected(received_coeffs),
                 'analysis_ms':analysis_ms,'forced_encode_ms':encode_ms,
                 'forced_encode_deadline_missed':encode_ms>PACKET_SECONDS*1000,
                 'parameter_normalized_max_abs_error':float(np.max(abs((expected-recovered)/wire.parameter_scale))),
                 'model_to_audio_display_rms_error':float(np.sqrt(np.mean((before-after)**2))),
                 'model_to_audio_raw_rms_error':float(np.sqrt(np.mean((before_raw-after_raw)**2))),
                 'forced_emitted_rms':float(np.sqrt(np.mean(audio**2))),
                 'existing_emitted_rms':float(np.sqrt(np.mean(ref_audio**2))),
                 'forced_emitted_peak':float(np.max(abs(audio))),
                 'existing_emitted_peak':float(np.max(abs(ref_audio)))}
            for prefix,image in (('existing',reference),('before',before),('after',after)):
                row.update({prefix+'_'+k:v for k,v in picture_metrics(original,image,True).items()})
            rows.append(row)
            w,h=size;comparison=Image.new('RGB',(4*w,h+30),'white');draw=ImageDraw.Draw(comparison)
            labels=('Original','Existing V7','Forced model BEFORE audio (diagnostic)','Forced model AFTER V7 audio (diagnostic)')
            folder=out/'images'/f'{profile}-{exposure.name}';folder.mkdir(parents=True)
            for i,(label,image) in enumerate(zip(labels,(original,reference,before,after))):
                picture=Image.fromarray(np.uint8(np.clip(image,0,1)*255))
                picture.save(folder/f'{i}.png');comparison.paste(picture,(i*w,30));draw.text((i*w+8,8),label,fill='black')
            comparison.save(folder/'comparison.png')
    json_write(out/'measurements.json',rows)
    json_write(out/'MODEL-DECODE-VALIDATION.json',{'passed':all(r['passed'] for r in model_checks),'checks':model_checks})
    summary={'pictures':len(rows),'mapping_passed':all(r['passed'] for r in mapping),
             'transport_control_passed':all(r['passed'] for r in controls),
             'model_decode_passed':all(r['passed'] for r in model_checks),'mapping_checks':mapping,'profiles':[]}
    for profile in PROFILES:
        group=[r for r in rows if r['profile']==profile]
        summary['profiles'].append({'profile':profile,'pictures':len(group),
            'adaptive_selections':sum(r['adaptive_would_select_geometry'] for r in group),
            'forced_branch_received':sum(r['diagnostic_geometry_selected_by_receiver'] for r in group),
            'before_ssim_delta':float(np.mean([r['before_ssim']-r['existing_ssim'] for r in group])),
            'after_ssim_delta':float(np.mean([r['after_ssim']-r['existing_ssim'] for r in group])),
            'model_to_audio_display_rms_error':float(np.mean([r['model_to_audio_display_rms_error'] for r in group])),
            'parameter_normalized_max_abs_error':max(r['parameter_normalized_max_abs_error'] for r in group),
            'forced_emitted_rms_mean_ratio':float(np.mean([r['forced_emitted_rms']/max(r['existing_emitted_rms'],1e-12) for r in group])),
            'forced_emitted_peak_max':max(r['forced_emitted_peak'] for r in group),
            'forced_encode_mean_ms':float(np.mean([r['forced_encode_ms'] for r in group])),
            'forced_encode_deadline_misses':sum(r['forced_encode_deadline_missed'] for r in group)})
    json_write(out/'SUMMARY.json',summary)
    lines=['# Encoder model versus audio diagnostic','',
           'Columns: Original / Existing V7 / Forced model BEFORE audio / Forced model AFTER V7 audio.',
           'The third column uses a noiseless production-fold model; only the second and fourth columns decode actual audio. '
           'The forced model is shown even when the adaptive encoder rejects it. This diagnoses failure, not a claimed resolution gain.','']
    lines += [f'![{r["profile"]} {r["asset"]}](images/{r["profile"]}-{r["asset"]}/comparison.png)\n' for r in rows]
    (out/'REPORT.md').write_text('\n'.join(lines))
    print('Completed source/transport diagnosis:',out/'SUMMARY.json')


if __name__=='__main__':main()
