"""Finite V7 runs, source transforms and deterministic offline WAV export."""
import itertools
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image


def validate_run_options(args):
    for name in ('frames', 'source_passes'):
        if getattr(args, name, 0) < 0:
            raise ValueError(f'--{name.replace("_", "-")} must be non-negative')
    seconds = getattr(args, 'seconds', 0)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('--seconds must be finite and non-negative')
    rate = getattr(args, 'rate', None)
    if rate is not None and rate <= 0:
        raise ValueError('--rate must be positive')
    if getattr(args, 'rotation', 0) not in (0, 90, 180, 270):
        raise ValueError('--rotation must be 0, 90, 180 or 270')
    if getattr(args, 'write', None) and getattr(args, 'record', None):
        raise ValueError('--write and --record are mutually exclusive')
    if getattr(args, 'source_passes', 0):
        from tools.v7_capture import is_file_video_source
        if (args.source != 'video' or getattr(args, 'video_live', False) or
                not is_file_video_source(args.video_source)):
            raise ValueError('--source-passes requires a local video file')
    if getattr(args, 'mono_sum', False) and getattr(args, 'channels', (0, 1)) != (0, 1):
        raise ValueError('--mono-sum uses one output channel; omit --channels')
    destination = getattr(args, 'write', None) or getattr(args, 'record', None)
    if destination and args.source == 'video':
        if Path(destination).expanduser().resolve() == Path(args.video_source).expanduser().resolve():
            raise ValueError('The output WAV must not overwrite the source video')


def transform_frame(frame, rotation=0, mirror=False):
    """Preserve capture geometry metadata while transforming the source RGB."""
    if not rotation and not mirror:
        return frame
    wrapped = hasattr(frame, 'source_size')
    rgb = frame.rgb if wrapped else frame
    data = np.asarray(rgb.convert('RGB') if isinstance(rgb, Image.Image) else rgb)
    if rotation:
        data = np.rot90(data, -(rotation // 90))
    if mirror:
        data = data[:, ::-1]
    data = np.ascontiguousarray(data)
    if wrapped:
        from tools.v7_capture import CapturedFrame
        size = frame.source_size
        if rotation in (90, 270):
            size = size[::-1]
        return CapturedFrame(data, size, frame.prepared)
    return data


class _ProcessReader:
    def __init__(self, command):
        root = Path(__file__).resolve().parents[1]/'tmp'
        root.mkdir(exist_ok=True)
        self.errors = tempfile.TemporaryFile(dir=root)
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                            stdin=subprocess.DEVNULL,
                                            stderr=self.errors)
        except BaseException:
            self.errors.close()
            raise

    def check(self):
        code = self.process.wait()
        if code:
            self.errors.seek(0)
            raise RuntimeError(self.errors.read().decode(errors='replace').strip()
                               or f'FFmpeg exited with status {code}')

    def close(self):
        self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.errors.close()


def _video_pass(source, fps, rate, start, soundtrack):
    """Yield one source picture per modem time step and its matching sound."""
    from tools.v7_capture import _read_ppm
    prefix = ['ffmpeg', '-nostdin', '-loglevel', 'error']
    if start:
        prefix += ['-ss', str(start)]
    prefix += ['-i', str(Path(source).expanduser())]
    video = _ProcessReader(prefix + ['-map', '0:v:0', '-vf', f'fps={fps:.12g}',
                                     '-pix_fmt', 'rgb24', '-c:v', 'ppm',
                                     '-f', 'image2pipe', '-an', '-sn', '-'])
    audio = None
    try:
        if soundtrack:
            probe = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=index', '-of', 'csv=p=0', str(source)],
                capture_output=True, text=True, check=True)
            if probe.stdout.strip():
                audio = _ProcessReader(prefix + ['-map', '0:a:0', '-vn', '-sn',
                                                 '-ac', '1', '-ar', str(rate),
                                                 '-f', 'f32le', '-'])
        while True:
            try:
                frame = _read_ppm(video.process.stdout)
            except RuntimeError:
                video.check()
                break
            yield frame, audio
    finally:
        video.close()
        if audio is not None:
            audio.close()


def export_wav(args, live=None):
    """Encode on a sample clock, never on wall time or an audio device."""
    if live is None:
        from tools import v7_live as live
    from tools.v7_capture import test_source
    from tools.v7_source_audio import PacketAudioDelay
    from tools.v7_wav_io import WavWriter

    validate_run_options(args)
    if args.source not in ('video', 'test') or getattr(args, 'video_live', False):
        raise ValueError('Offline export supports local video files and test sources')
    if args.source == 'video':
        from tools.v7_capture import is_file_video_source
        if not is_file_video_source(args.video_source):
            raise ValueError('Offline export requires a local video file')
    if getattr(args, 'source_audio', None) == 'device':
        raise ValueError('Offline export cannot capture a live audio input')
    if args.source == 'test' and not (args.frames or args.seconds):
        raise ValueError('Test-source export requires --frames or --seconds')
    encoder = live.build_send_encoder(args)
    encoder.warmup()
    rate = int(args.rate or live.P.RATE)
    if not live.P.MIN_PLAYBACK_SPEED <= args.speed <= live.P.MAX_PLAYBACK_SPEED:
        raise ValueError('Playback speed out of range')
    fps = live.FPS*args.speed
    limit = args.frames or (math.ceil(args.seconds*fps) if args.seconds else 0)
    if args.frames and args.seconds:
        limit = min(limit, math.ceil(args.seconds*fps))
    passes = args.source_passes or (1 if not limit else 0)
    total = 0
    packet_delay = PacketAudioDelay(encoder.extra_audio_delay_ms)
    clock = [0.0]
    synthetic = test_source(clock=lambda: clock[0])
    with WavWriter(args.write, rate, 1 if args.mono_sum else 2) as sink:
        pass_index = 0
        while not passes or pass_index < passes:
            if args.source == 'video':
                source = _video_pass(args.video_source, fps, rate,
                                     getattr(args, 'video_start', 0) if pass_index == 0 else 0,
                                     encoder.mono_fold_profile and encoder.source_audio_mode == 'source')
            else:
                source = ((synthetic(), None) for _ in itertools.count())
            before = total
            try:
                for frame, soundtrack in source:
                    if getattr(args, 'stop_event', None) is not None and args.stop_event.is_set():
                        return total
                    gains = []
                    value, aspect = encoder.frame_values(frame, deferred_gains_out=gains)
                    wire = encoder.wire_audio([value], [aspect], total+1,
                                              [live._wire_index(total+1)], gains or None)
                    if args.mono_sum:
                        wire = (wire[:, :1].copy() if encoder.mono_wire is not None
                                else wire.sum(axis=1, keepdims=True)/np.sqrt(2))
                        peak = np.max(np.abs(wire))
                        if peak > .89:
                            wire *= .89/peak
                    samples = live.P.speed_pulse_stream(wire, args.speed, rate=rate)
                    if encoder.mono_fold_profile:
                        count = len(samples)
                        sound = np.zeros(count, np.float32)
                        if soundtrack is not None:
                            data = soundtrack.process.stdout.read(count*4)
                            if data:
                                sound[:len(data)//4] = np.frombuffer(data, '<f4')
                            if len(data) < count*4:
                                soundtrack.check()
                        sound = packet_delay.apply(
                            np.clip(sound*encoder.audio_gain, -1.0, 1.0), count, rate)
                        samples[:, 1-encoder.mono_wire.carrier_index] = sound
                    sink.write(samples)
                    total += 1
                    clock[0] = total/fps
                    if limit and total >= limit:
                        return total
            finally:
                source.close()
            if total == before:
                raise ValueError('Video source produced no pictures')
            pass_index += 1
    return total


def prime_file_profile(args, decoder, live):
    """Look ahead for a confirmed startup profile, then rewind for decoding.

    Live reception can discard acquisition packets. A file can look ahead and
    replay them instead, using the same status-confirmation policy and the
    existing startup-hint path (each decoded packet still validates its wire).
    """
    from animation_modem.audio_common import input_pair
    from tools.v7_wav_io import wav_blocks, wav_info
    rate, channels, _ = wav_info(args.read)
    selected = input_pair(getattr(args, 'channels', None), channels)
    decoder.input_channels = len(selected)
    decoder.capture_rate = rate
    probes = [live._ProfileStatusProbe(rate, index, direction=args.direction)
              for index in range(len(selected))]
    position = 0
    hint = None
    for block in wav_blocks(args.read):
        if getattr(args, 'stop_event', None) is not None and args.stop_event.is_set():
            break
        events = []
        position += len(block)
        for index, probe in enumerate(probes):
            probe.add(block[:, selected[index]:selected[index]+1])
            events.extend(probe.scan(position/rate))
        decisions = decoder.observe_packets(events, now=position/rate)
        if any(decision['confirmed'] for decision in decisions):
            hint = decoder.profile_name, decoder.active_side
            break
    decoder.reset_capture_timeline()
    if hint is not None:
        decoder.start_with(*hint)
