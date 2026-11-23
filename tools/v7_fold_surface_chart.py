"""Descending analog packing ratios versus actual decoded target passes."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tools.v7_color_gain_loss_chart import font
from tools.v7_color_resolution_bench import json_write


def write_chart(out,rows):
    out=Path(out)
    summaries=[]
    configs=sorted({r['config'] for r in rows if r['config'].startswith('surface-')},
                   key=lambda c:float(c.split('-')[1].split('to')[0])/float(c.split('-')[1].split('to')[1]),reverse=True)
    configs=['default']+configs
    targets=sorted({k for r in rows for k in r if k.startswith('display_') and k.endswith('_resolved')})
    for profile in sorted({r['profile'] for r in rows}):
        width=540+105*len(targets)
        image=Image.new('RGB',(width,130+40*len(configs)),'#f6f7f8');draw=ImageDraw.Draw(image)
        draw.text((16,12),profile+' — aggressive packing down to identity',font=font(20,True),fill='#17212b')
        draw.text((16,44),'Cells: passing movie-frame fraction. More packed coordinates is not a resolution claim.',font=font(14),fill='#53606c')
        for j,k in enumerate(targets):
            draw.text((540+105*j,78),k.removeprefix('display_').removesuffix('_resolved').replace('horizontal','H').replace('vertical','V'),font=font(12),fill='#17212b')
        for i,config in enumerate(configs):
            group=[r for r in rows if r['profile']==profile and r['config']==config]
            movies=[r for r in group if r['asset'].startswith('v7_pixel_motion')]
            y=105+i*40
            draw.text((16,y+9),config,font=font(13),fill='#17212b')
            overload=[r['source_range_overload_fraction'] for r in group if 'source_range_overload_fraction' in r]
            extra=[r['extra_source_coordinates'] for r in group if 'extra_source_coordinates' in r]
            draw.text((280,y+9),f"shown {sum(r.get('shown',False) for r in group)}/{len(group)}"+('' if not overload else f'  clip {np.mean(overload):.1%}'),font=font(12),fill='#17212b')
            record={'profile':profile,'config':config,'trials':len(group),'errors':sum('error' in r for r in group),
                    'carrier_headroom':next((r['carrier_headroom'] for r in group if 'carrier_headroom' in r),None),
                    'mean_source_overload':float(np.mean(overload)) if overload else None,
                    'extra_coordinates':extra,'passing':{}}
            for j,k in enumerate(targets):
                passed=sum(bool(r.get(k,False)) for r in movies)
                fraction=passed/len(movies) if movies else 0
                x=540+j*105;draw.rectangle((x,y,x+100,y+36),fill=(int(245-45*fraction),int(215+25*fraction),215))
                draw.text((x+5,y+10),f'{passed}/{len(movies)}',font=font(13),fill='#17212b')
                record['passing'][k]={'passed':passed,'tested':len(movies)}
            summaries.append(record)
        image.save(out/f'PACKING-{profile}.png')
    json_write(out/'packing-summary.json',summaries)
    report=out/'REPORT.md'
    with report.open('a') as stream:
        stream.write('\n## Descending capacity sweep\n\n')
        stream.write('High dimension ratios are stress tests, not predicted resolution gains. '
                     'Source overload and unavailable/error trials remain explicit. '
                     'All cases use clean actual V7 packets.\n\n')
        for p in sorted(out.glob('PACKING-*.png')):
            stream.write(f'![{p.stem}]({p.name})\n\n')
