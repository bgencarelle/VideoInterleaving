"""VideoInterleaving clock/selector -> baked RGBA composite -> stereo modem.

Transmits with the v2 transport. The shared-time path is unchanged: v2's header
keeps a 32-bit millisecond field, so reserve() still projects a send deadline,
the image index is still evaluated at that instant, and the receiver can still
hold the frame until then. Only the encoder underneath changed.
"""
import json
from pathlib import Path
import time
import wave
from animation_modem.transport2 import PRESETS, RATE, SourceCoder, encode
from animation_modem.imaging import DEFAULT_PROFILE, fit_shapes, image_values, plane_shapes
from animation_modem.audio_common import device, pair, pcm
from animation_modem.playback import PacketOutput, latency
from modem_bake import ModemLibrary


def fixed_pair(value):
    if value is None:return None
    try:
        result=tuple(int(v) for v in value.split(','))
        if len(result)!=2 or min(result)<0:raise ValueError()
        return result
    except ValueError:
        raise ValueError('--modem-pair needs zero-based face,float indices, e.g. 1,0')


def packet(library, layout, coder, absolute, selection, numbered=False,
           background=(4,4,4), rotation=0, mirror=False, target_time_ns=None):
    import time as _time
    index,main,front=selection
    started=_time.perf_counter()
    im=library.composite(index,main,front,background,rotation,mirror)
    if numbered:
        im=burn_counters(im,absolute,index+1,library.frames)
    values=image_values(im,coder.shapes)
    # v2 identifies frames on a 16-bit index/count, so a long bake wraps rather
    # than raising. The absolute frame number stays 32-bit.
    stamp_ms=0 if target_time_ns is None else (int(target_time_ns)//1_000_000) & 0xffffffff
    audio=encode(values,layout,coder,absolute & 0xffffffff,
                 (index % 0xffff)+1,max(1,min(library.frames,0xffff)),stamp_ms=stamp_ms)
    ms=(_time.perf_counter()-started)*1000
    return audio, {'frame':absolute,'source_index':index,'face_folder':main,
                   'float_folder':front,'layout':layout.name,'encode_ms':ms,
                   'target_time_ns':target_time_ns}, im


def burn_counters(im, absolute, index, count):
    """Debug overlay, kept out of the transport so the wire format stays fixed."""
    from PIL import ImageDraw
    im=im.copy()
    draw=ImageDraw.Draw(im)
    draw.text((1,1),f'A{absolute}',fill='white')
    draw.text((1,9),f'F{index}/{count}',fill='white')
    return im


def run_modem(args):
    import settings
    root=args.modem_dir or getattr(settings,'MODEM_DIR',settings.IMAGES_DIR+'_modem')
    library=ModemLibrary(root)
    # The wire format is fixed and independent of the bake's own profile; the
    # bake profile only affects how much source detail exists to send.
    layout=PRESETS[getattr(args,'modem_preset',None) or 'wide']
    coder=SourceCoder(fit_shapes(plane_shapes(DEFAULT_PROFILE),layout.capacity))
    selected=fixed_pair(args.modem_pair)
    if selected is not None:
        if selected[0]>=len(library.mains) or selected[1]>=len(library.floats):
            raise ValueError('--modem-pair is outside the selected baked folder lists')
    elif len(library.mains)<2:
        raise ValueError('The live selector needs a rest folder plus another face folder; use --modem-pair 0,0 for one pair')
    if args.modem_wav and selected is None:
        raise ValueError('--modem-wav requires --modem-pair, e.g. 1,0; runtime stochastic selection is not prebaked')
    channels=pair(args.modem_channels)
    output_latency=latency(args.modem_latency)
    background=getattr(settings,'BACKGROUND_COLOR',(4,4,4))
    rotation=args.rotation if args.rotation is not None else getattr(settings,'INITIAL_ROTATION',0)
    mirror=args.mirror if args.mirror is not None else bool(getattr(settings,'INITIAL_MIRROR',0))
    print(f'[MODEM] {root}: {library.frames} images, {len(library.mains)} face / '
          f'{len(library.floats)} float folders, bake profile {library.profile}; '
          f'source {settings.IPS} ips, transport {layout.fps:.2f} fps at {RATE} Hz stereo')
    print(f'[MODEM] {layout.describe()}')
    print('[MODEM] Face order:',[str(p.relative_to(library.root)) for p in library.mains])
    print('[MODEM] Float order:',[str(p.relative_to(library.root)) for p in library.floats])
    if args.modem_wav:
        limit=args.modem_frames or library.frames
        path=Path(args.modem_wav)
        path.parent.mkdir(parents=True,exist_ok=True)
        with wave.open(str(path),'wb') as sink:
            sink.setparams((2,2,RATE,0,'NONE','not compressed'))
            for n in range(limit):
                # Inspection export: deterministic forward source order at 15 fps.
                audio,report,_=packet(library,layout,coder,n+1,
                                      (n%library.frames,*selected),
                                      args.modem_numbered,background,rotation,mirror)
                sink.writeframesraw(pcm(audio))
                if args.modem_log_frames:print(json.dumps(report),flush=True)
        print(f'[MODEM] Wrote {limit} independent frames to {path}')
        return
    # Use the same clock and stateful folder selector as image_display.run_display
    # and scope_display.run_scope. There is no separate modem clock/thread.
    import index_calculator
    from folder_selector import update_folder_selection, folder_dictionary
    index_calculator.set_clock_mode(args.modem_clock)
    index_calculator.png_paths_len=library.frames
    index_calculator.frame_duration=args.modem_frame_duration
    if index_calculator.midi_mode and args.modem_time_offset_ms:
        raise ValueError('--modem-time-offset-ms requires the free clock; MIDI events cannot be predicted')
    prepare_ms=args.modem_prepare_ms
    receive_margin_ms=args.modem_receive_margin_ms + args.modem_time_offset_ms
    # Timing offsets move the presentation instant AND the index together, so they
    # cannot correct an index that disagrees with the other modes. This shifts only
    # which image the clock formula returns, leaving the shared epoch and the
    # transmitted timestamp alone.
    index_offset_ns=round(getattr(args,'modem_index_offset_ms',0.0)*1_000_000)
    previous=None
    n=0
    # Poll at the existing display rate; packet readiness, not sleep, controls
    # transmission. Unchanged source indices can still produce complete packets.
    poll_seconds=1 / (settings.FPS or 60)
    with PacketOutput(device(args.scope_device),channels,output_latency,
                      frame=layout.frame,packet=layout.packet) as output:
        while not args.modem_frames or n < args.modem_frames:
            started=time.perf_counter()
            index,_=index_calculator.update_index(library.frames,settings.PINGPONG)
            index=max(0,min(int(index),library.frames-1))
            if selected is None:
                if index != previous:
                    update_folder_selection(index,len(library.floats),len(library.mains))
                folders=folder_dictionary['Main_and_Float_Folders']
            else:
                folders=selected
            previous=index
            # Scope's ready()/pending pattern: no generator lookahead, no FIFO
            # of stale images, and never replace a partially transmitted packet.
            if output.ready():
                slot=None
                target_time_ns=None
                if not index_calculator.midi_mode:
                    slot=output.reserve(prepare_ms,receive_margin_ms)
                    target_time_ns=slot.target_time_ns
                    # Same clock formula as every mode; do not mutate the live
                    # selector's state while looking up a future image index.
                    index,_=index_calculator.calculate_free_clock_index(
                        library.frames,settings.PINGPONG,at_time_ns=target_time_ns,
                        time_offset_ns=index_offset_ns,publish=False)
                encode_started=time.perf_counter()
                audio,report,_=packet(library,layout,coder,n+1,(index,*folders),
                                      args.modem_numbered,background,rotation,mirror,
                                      target_time_ns)
                elapsed_ms=(time.perf_counter()-encode_started)*1000
                # Extra preparation time changes the send deadline, not the
                # animation epoch. Slow machines skip stale work and retry.
                prepare_ms=max(args.modem_prepare_ms,elapsed_ms*1.5)
                accepted=output.submit(audio,slot)
                if accepted:
                    n+=1
                    if args.modem_log_frames:print(json.dumps(report),flush=True)
            delay=poll_seconds-(time.perf_counter()-started)
            if delay > 0:time.sleep(delay)
        output.finish()
