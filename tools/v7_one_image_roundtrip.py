"""One actual V7 packet, independent receiver, saved image outputs in tmp/."""
from pathlib import Path
import sys
import json
import wave
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from numba import njit
from PIL import Image,ImageDraw
from tools.v7_patch_residual import PatchResidualWire
from tools.v7_analog_frame import ProductionFrameWire
from tools.v7_analog_frame_bench import first_picture
from tools.v7_color_corpus import face_image,face_path
from tools.v7_color_metrics import resize_rgb,render as production_render
from tools.v7_local_detail_metrics import unit_image,image_bytes,audio_levels,attenuate
from animation_modem.v7_core import adapt_packet_for_output

out=ROOT/'tmp/v7-one-image-roundtrip'
out.mkdir(parents=True,exist_ok=True)
profile='aspect-fold-500';layout='3:4';index=1400;size=(576,768)
source=face_image(index)
sender=PatchResidualWire(profile,layout)
values=sender.values(source)
diagnostics=sender.last_diagnostics.copy()
original=resize_rgb(unit_image(source),size)
Image.fromarray(image_bytes(original)).save(out/'original.png')
if not diagnostics['geometry_selected']:
    raise RuntimeError('Source reconstruction rejected; no experimental packet generated')
baseline=ProductionFrameWire(profile,layout)
reference=adapt_packet_for_output(baseline.encode_packet(source,1,index),96000)
reference_rms,reference_peak=audio_levels(reference)
# Encode these exact converter values once; do not run the source fitter again.
sender.values=lambda _:values
audio=adapt_packet_for_output(sender.encode_packet(source,1,index),96000)
audio,gain=attenuate(audio,reference_rms,reference_peak)
receiver=PatchResidualWire(profile,layout)
result,received,_=first_picture(receiver,audio,index)
baseline_receiver=ProductionFrameWire(profile,layout)
baseline_result,baseline_received,_=first_picture(baseline_receiver,reference,index)
if received is None or baseline_received is None:
    raise RuntimeError('A packet failed to produce a decoded image')
decoded=receiver.render_received(received,size,False)
existing=production_render(baseline_receiver,baseline_received,size,False)
for name,image in (('decoded-experimental.png',decoded),('decoded-existing-v7.png',existing)):
    Image.fromarray(image_bytes(image)).save(out/name)
canvas=Image.new('RGB',(3*size[0],size[1]+42),'white');draw=ImageDraw.Draw(canvas)
for i,(label,image) in enumerate((('Original',original),('Existing V7: decoded audio',existing),
    ('Experimental source-aware V7: decoded audio',decoded))):
    draw.text((i*size[0]+8,12),label,fill='black')
    canvas.paste(Image.fromarray(image_bytes(image)),(i*size[0],42))
canvas.save(out/'comparison.png')
rms,peak=audio_levels(audio)
with wave.open(str(out/'encoded.wav'),'wb') as file:
    file.setnchannels(2);file.setsampwidth(2);file.setframerate(96000)
    # Saved PCM is illustrative; independent decoding above uses original
    # floating-point waveform, avoiding a second quantized channel test.
    @njit
    def pcm(samples):
        return np.rint(np.minimum(1.,np.maximum(-1.,samples))*32767).astype(np.int16)
    file.writeframes(pcm(audio).tobytes())
coefficients=receiver.codec.grid.forward(received)
summary={'source':str(face_path(index)),'source_index':index,'profile':profile,
    'converter':'PatchResidualWire','output_size':list(size),'actual_audio_decode':True,
    'independent_receiver':True,'source_selected':diagnostics['geometry_selected'],
    'experimental_branch_received':receiver.geometry_selected(coefficients),
    'received_status':result.status,'existing_received_status':baseline_result.status,
    'uniform_volume_gain':gain,'rms':rms,'peak':peak,
    'reference_rms':reference_rms,'reference_peak':reference_peak,
    'resolution_gain_claim':False,'source_diagnostics':diagnostics}
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
