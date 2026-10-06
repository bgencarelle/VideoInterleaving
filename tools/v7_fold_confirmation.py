"""Compact completed-run outcomes; retain per-trial records without streaming."""
import json
from pathlib import Path
import numpy as np
from tools.v7_color_resolution_bench import json_write


def write_outcome(out,rows):
    out=Path(out)
    control_path=out/'CONTROL-VALIDATION.json'
    control=json.loads(control_path.read_text()) if control_path.exists() else None
    summary={'trials':len(rows),'errors':sum('error' in r for r in rows),
             'pictures':sum(r.get('shown',False) for r in rows),
             'fidelity_gate_passed':control['passed'] if control else None,
             'max_control_error':max((r['decoded_max_abs_error'] or 0 for r in control['trials']),default=0) if control else None,
             'loop_trials':sum('loop_packets' in r for r in rows),
             'loop_failures':sum(r.get('loop_independent_shown')!=r['loop_packets'] or
                                 r.get('loop_continuous_shown')!=r['loop_packets'] for r in rows if 'loop_packets' in r),
             'comparisons':[]}
    baselines={(r['profile'],r['asset'],r['case']):r for r in rows if r['config']=='default'}
    for profile,config in sorted({(r['profile'],r['config']) for r in rows}):
        group=[r for r in rows if r['profile']==profile and r['config']==config]
        item={'profile':profile,'config':config,'pictures':sum(r.get('shown',False) for r in group),'trials':len(group),
              'directional_frame_gains':{axis:{'1.5x':0,'2x':0,'3x':0,'comparable':0,'both_unresolved':0} for axis in ('horizontal','vertical')}}
        for metric in ('display_ssim','display_delta_e76','display_hair_band1_correlation','emitted_rms'):
            delta=[]
            for r in group:
                b=baselines.get((profile,r['asset'],r['case']),{})
                if r.get(metric) is not None and b.get(metric) is not None:
                    delta.append(r[metric]-b[metric])
            item[metric+'_paired_mean_delta']=float(np.mean(delta)) if delta else None
        for r in group:
            if not r['asset'].startswith('v7_pixel_motion'):
                continue
            base=baselines.get((profile,r['asset'],r['case']),{})
            for axis in ('horizontal','vertical'):
                old=[n for n in (2,3,4,6,8) if base.get(f'display_{axis}_pitch{n}_resolved')]
                new=[n for n in (2,3,4,6,8) if r.get(f'display_{axis}_pitch{n}_resolved')]
                d=item['directional_frame_gains'][axis]
                if old:
                    d['comparable']+=1
                    ratio=min(old)/min(new) if new else 0.
                    for label,threshold in (('1.5x',1.5),('2x',2.),('3x',3.)):
                        d[label]+=int(ratio>=threshold)
                elif not new:
                    d['both_unresolved']+=1
        summary['comparisons'].append(item)
    json_write(out/'OUTCOME.json',summary)
    (out/'OUTCOME.md').write_text(
        '# Computed completed-run outcome\n\n'+
        f"Trials: {summary['trials']}; pictures: {summary['pictures']}; errors: {summary['errors']}.\n\n"+
        f"Fidelity gate: {summary['fidelity_gate_passed']}; loop failures: {summary['loop_failures']}.\n\n"+
        'Fixture gains are conditional per-frame/direction comparisons, not general spatial-resolution claims. '
        'Dense-probe limits and held-out examples must confirm the acceptance target.\n')
    return summary
