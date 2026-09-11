"""Generate or observe the experimental two-channel timing/pilot reference."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.io import wavfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animation_modem.pilot_clock_prototype import PilotClock, reference


def summary(records, seconds, elapsed, max_call):
    if not records:
        return dict(status='no_reference_lock', audio_seconds=seconds, cpu_ms=elapsed*1000)
    fields = ('delivery_speed', 'frequency_scale', 'extra_pitch_cents', 'frequency_offset_hz')
    return dict(status='reference_tracked', reports=len(records),
                **{k:float(np.median([r[k] for r in records])) for k in fields},
                audio_seconds=seconds, cpu_ms=elapsed*1000, max_feed_ms=max_call*1000,
                acquisitions=records[-1]['acquisitions'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    write=sub.add_parser('write')
    write.add_argument('--out',type=Path,default=Path('pilot_reference.wav'))
    write.add_argument('--seconds',type=float,default=4.)
    write.add_argument('--speed',type=float,default=1.)
    write.add_argument('--pitch-cents',type=float,default=0.)
    write.add_argument('--wow-percent',type=float,default=0.)
    write.add_argument('--wow-hz',type=float,default=4.)
    write.add_argument('--offset-hz',type=float,default=0.)
    read=sub.add_parser('read')
    read.add_argument('--wav',type=Path,required=True)
    read.add_argument('--verbose',action='store_true')
    live=sub.add_parser('live-receive')
    live.add_argument('--device',type=lambda s:int(s) if s.isdecimal() else s)
    live.add_argument('--channels',default='1,2')
    args=parser.parse_args()
    if args.command=='write':
        audio=reference(args.seconds,speed=args.speed,cents=args.pitch_cents,
                        wow_percent=args.wow_percent,wow_hz=args.wow_hz,offset_hz=args.offset_hz)
        wavfile.write(args.out,48000,np.rint(audio*32767).astype(np.int16))
        print(f'Wrote {args.out}: L timing marks, R pilots; reference only, no images.')
    elif args.command=='read':
        rate,audio=wavfile.read(args.wav)
        if audio.dtype!=np.int16:
            parser.error('Use a stereo 16-bit PCM WAV')
        audio=audio.astype(np.float32)/32768
        rx=PilotClock(rate);records=[];max_call=0.;started=time.perf_counter()
        for at in range(0,len(audio),256):
            t=time.perf_counter();got=rx.feed(audio[at:at+256])
            max_call=max(max_call,time.perf_counter()-t)
            records.extend(got)
            if args.verbose:
                for row in got:print(json.dumps(row))
        print(json.dumps(summary(records,len(audio)/rate,time.perf_counter()-started,max_call)))
    else:
        import sounddevice as sd
        channels=tuple(int(c)-1 for c in args.channels.split(','))
        if len(channels)!=2 or min(channels)<0 or channels[0]==channels[1]:
            parser.error('Choose two distinct positive channel numbers')
        info=sd.query_devices(args.device,'input')
        rx=PilotClock(info['default_samplerate']);last_report=0.
        try:
            with sd.InputStream(device=args.device,samplerate=rx.rate,channels=max(channels)+1,
                                dtype='float32',blocksize=0) as stream:
                while True:
                    audio,overflow=stream.read(256)
                    if overflow:rx.reset()
                    got=rx.feed(audio[:,channels])
                    now=time.monotonic()
                    if got and now-last_report>=1:
                        print(json.dumps(got[-1]),flush=True);last_report=now
        except KeyboardInterrupt:
            pass


if __name__=='__main__':
    main()
