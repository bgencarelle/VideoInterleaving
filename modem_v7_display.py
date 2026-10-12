"""V7 pulse-wire sender using the application's shared image clock.

This is the application adapter around the V7 transport. It owns source
composition and scheduling, and its ``encode_values_packet`` boundary is the
production fold/pilot/packet path shared by offline image-quality trials.
"""
import json
import math
import sys
from pathlib import Path
import time
import wave

import numpy as np

from animation_modem.audio_common import device, pair, pcm
from animation_modem.v7_core import speed_length, speed_resample
from animation_modem.playback import PacketOutput, latency
from modem_bake import ModemLibrary
from animation_modem import v7 as _v7
from animation_modem.v7_coded_pilot import add_fold500_coded_pilot


TARGET_RMS = .1521


def _wav_packet_limit(frames, speed, ips, pingpong, packet_count=0,
                      cycles=None):
    """Return the finite packet count for an offline IPS-paced WAV."""
    if packet_count:
        return int(packet_count)
    span = (_v7.loop_period(frames, pingpong) * int(cycles)
            if cycles is not None else frames)
    return max(1, math.ceil(span * _v7.PULSE_FPS * speed / float(ips)))


def fixed_pair(value):
    if value is None:
        return None
    try:
        result = tuple(int(v) for v in value.split(','))
        if len(result) != 2 or min(result) < 0:
            raise ValueError()
        return result
    except ValueError:
        raise ValueError('--modem-pair needs zero-based face,float indices, e.g. 1,0')


def _source_values(model, image, encode_filter='nearest'):
    prepared = _v7.prepare_image(image, encode_filter=encode_filter)
    return _v7.image_values(prepared, model.coder.grids,
                            encode_filter=encode_filter)


def _source_dct_coefficients(model, image, encode_filter='box'):
    """Convert an image straight to the production coder-grid DCT vector.

    This follows the same resize, Pillow YCbCr conversion, and per-plane BOX
    sampling as ``_source_values``. It skips building and flattening the
    intermediate spatial value vector, then returns the full-grid DCT that
    Fold 500 consumes. This is the fast, exact application-grid path; it is
    separate from the experimental native-resolution DCT projections.
    """
    if encode_filter != 'box':
        raise ValueError('direct image DCT requires the Box encode filter')
    from PIL import Image
    from scipy.fft import dctn

    prepared = _v7.prepare_image(image, encode_filter='box')
    planes = prepared.convert('YCbCr').split()
    coefficients = []
    for plane, (rows, cols) in zip(planes, model.coder.grids):
        if plane.size != (cols, rows):
            plane = plane.resize((cols, rows), Image.Resampling.BOX)
        values = np.asarray(plane, dtype=np.float64)/127.5 - 1.0
        coefficients.append(dctn(values, norm='ortho').ravel())
    return np.concatenate(coefficients)


def encode_values_packet(model, values, absolute, source_index, aspect_code, *,
                         loop=None, direction=1, pilot_tones=True, fold=None):
    """Encode prepared values with the application's production V7 packet path.

    Kept separate from compositing so offline image-quality tests can feed
    candidate value vectors through the exact production fold, packet, and
    coded-pilot implementation.
    """
    if fold is not None and int(model.encoding_type) != \
            _v7.ENCODING_FILTER_CODES['box']:
        raise ValueError('stereo Fold 500 requires the Box encode filter')
    if fold is None:
        return _v7.encode_pulse_frame(
            model, values, absolute, aspect_code=aspect_code,
            source_index=source_index, loop=loop, direction=direction,
            pilot_tones=pilot_tones)
    coefficients = fold.encode_coefficients(values)
    return encode_folded_coefficients_packet(
        model, coefficients, absolute, source_index, aspect_code,
        loop=loop, direction=direction)


def encode_image_dct_packet(model, image, absolute, source_index, aspect_code, *,
                            loop=None, direction=1,
                            fold=None, encode_filter='box'):
    """Opt-in Box image path that sends coder-grid DCTs through Fold 500."""
    if fold is None:
        raise ValueError('image-to-DCT packet encoding requires Fold 500')
    if int(model.encoding_type) != _v7.ENCODING_FILTER_CODES['box']:
        raise ValueError('stereo Fold 500 requires the Box encode filter')
    coefficients = fold.encode_dct_coefficients(
        _source_dct_coefficients(model, image, encode_filter))
    return encode_folded_coefficients_packet(
        model, coefficients, absolute, source_index, aspect_code,
        loop=loop, direction=direction)


def encode_folded_source_packet(model, rgb, absolute, source_index, aspect_code, *,
                                fold, loop=None, direction=1):
    """Opt-in native RGB projection into the exact Fold-500 source slots."""
    if fold is None:
        raise ValueError('native source projection requires Fold 500')
    if int(model.encoding_type) != _v7.ENCODING_FILTER_CODES['box']:
        raise ValueError('stereo Fold 500 requires the Box encode filter')
    from animation_modem.v7_source_dct import source_fold_dct_coefficients

    full = source_fold_dct_coefficients(
        rgb, _v7.V7_GRIDS, fold.source_positions)
    coefficients = fold.encode_dct_coefficients(full)
    return encode_folded_coefficients_packet(
        model, coefficients, absolute, source_index, aspect_code,
        loop=loop, direction=direction)


def encode_folded_coefficients_packet(
        model, coefficients, absolute, source_index, aspect_code, *,
        loop=None, direction=1):
    """Transmit coefficients already mapped through the production Fold 500.

    This is the same pulse/coded-pilot path used by ``encode_values_packet``;
    the coefficient-domain entry lets source-DCT experiments apply Fold 500
    directly without a DCT -> pixels -> DCT detour.
    """
    audio = _v7.encode_pulse_frame_coeffs(
        model, coefficients, absolute,
        aspect_code=aspect_code, source_index=source_index,
        loop=loop, direction=direction, pilot_tones=False,
        extra_tone_mixer=lambda packet: add_fold500_coded_pilot(
            packet, absolute))
    return audio


def packet(library, model, absolute, source_index, selection, *,
           background=(4, 4, 4), rotation=0, mirror=False,
           encode_filter='nearest', loop=None, direction=1,
           pilot_tones=True, fold=None):
    if fold is not None and encode_filter != 'box':
        raise ValueError('stereo Fold 500 requires the Box encode filter')
    index, main_folder, float_folder = selection
    if encode_filter == 'nearest' and hasattr(library, 'composite_nearest'):
        # Nearest sampling reads 80x96 source pixels: composite only those
        # (bit-identical; full-size compositing was most of the encode time).
        image = library.composite_nearest(index, main_folder, float_folder,
                                          _v7.PREPARED_SIZE, background,
                                          rotation, mirror)
    else:
        image = library.composite(index, main_folder, float_folder,
                                  background, rotation, mirror)
    values = _source_values(model, image, encode_filter)
    aspect_code = _v7.aspect_wire_code(
        image.info.get('source_dimensions', image.size))
    audio = encode_values_packet(
        model, values, absolute, source_index, aspect_code,
        loop=loop, direction=direction, pilot_tones=pilot_tones, fold=fold)
    return audio, {
        'frame': absolute,
        'source_index': source_index,
        'face_folder': main_folder,
        'float_folder': float_folder,
        'aspect_code': aspect_code,
        'packet_samples': len(audio),
        'peak': float(np.max(np.abs(audio))) if audio.size else 0.0,
        'pcm_clip_samples': int(np.count_nonzero(np.abs(audio) >= 1.0)),
        'limiter_active': False,
        'fold_slots': 500 if fold is not None else 0,
    }


# Packets whose pictures are loaded and encoded ahead of the one being sent.
AHEAD_PACKETS = 3


def send_args(args):
    """The V7 sender's arguments for ``main.py --mode modem``.

    Nothing but what the application was asked for: every other choice is
    the sender's shipped default (tools/v7_send_defaults.py), the same one an
    untouched sender GUI starts with, so a new best default reaches both.
    """
    from tools import v7_live
    # The encode opens no device; the name only satisfies the parser.
    argv = ['send', '--device', 'main.py']
    if getattr(args, 'modem_profile', None):
        argv += ['--profile', args.modem_profile]
    if getattr(args, 'modem_pixel_encode', False):
        argv.append('--pixel-encode')
    argv += list(getattr(args, 'modem_send_arg', None) or ())
    return v7_live.parser().parse_args(argv)


def build_encoder(args):
    """The V7 sender's picture encode and wire (v7_live.build_send_encoder)."""
    from tools import v7_live
    return v7_live.build_send_encoder(send_args(args))


def encoded_picture(encoder, library, index, main_folder, float_folder, *,
                    background=(4, 4, 4), rotation=0, mirror=False):
    """(values, aspect code): one composited picture, loaded and encoded
    exactly as the V7 sender encodes a frame. The slow part of a packet."""
    from tools.v7_capture import CapturedFrame
    image = library.composite(index, main_folder, float_folder,
                              background, rotation, mirror)
    # A bake can be smaller than its source: the aspect code is named from
    # the source picture's size, as the sender names it from a capture's.
    frame = CapturedFrame(
        np.asarray(image.convert('RGB')),
        tuple(image.info.get('source_dimensions', image.size)))
    return encoder.frame_values(frame)


def wire_packet(encoder, value, aspect_code, absolute, source_index, *,
                loop=None, direction=1):
    """One 48 kHz packet of an encoded picture, numbered ``absolute`` and
    carrying ``source_index`` and the loop. The quick part (about 2 ms)."""
    stamps = {} if loop is None else {int(source_index): (loop, direction)}
    with _v7.send_stamps(stamps):
        return encoder.wire_audio([value], [aspect_code], absolute,
                                  [int(source_index)])


class PicturesAhead:
    """Loads and encodes pictures before their send time, so the send loop
    only hands a finished one to the wire when its slot comes.

    The application's local mode does the same for its window: workers load
    upcoming images into a buffer and the display takes the nearest one that
    is ready, never waiting. Here each worker job loads, composites and
    encodes one picture together. A packet carries the index of the picture
    it really holds, so the receiver's loop readout stays true.
    """

    def __init__(self, make, frames, workers=2, keep=8):
        from collections import OrderedDict
        from concurrent.futures import ThreadPoolExecutor
        self._make = make
        self.frames = max(1, int(frames))
        self.keep = int(keep)
        self._jobs = OrderedDict()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix='v7-modem-ahead')
        self.exact = self.nearest = self.waited = 0

    def request(self, index, folders):
        key = (int(index), tuple(int(v) for v in folders))
        if key in self._jobs:
            self._jobs.move_to_end(key)
            return
        self._jobs[key] = self._pool.submit(self._make, *key)
        while len(self._jobs) > self.keep:
            _old, future = self._jobs.popitem(last=False)
            future.cancel()                    # not started yet: skip it

    def _distance(self, a, b):
        d = abs(int(a)-int(b)) % self.frames
        return min(d, self.frames-d)

    def take(self, index, folders):
        """(index sent, (values, aspect)): the picture for ``index`` if it is
        finished, else the nearest finished one for these folders; waits only
        when nothing at all is finished (the first packet, a folder change)."""
        folders = tuple(int(v) for v in folders)
        key = (int(index), folders)
        exact = self._jobs.get(key)
        if exact is not None and exact.done() and not exact.cancelled():
            self.exact += 1
            return key[0], exact.result()
        ready = [(self._distance(i, index), i, future)
                 for (i, f), future in self._jobs.items()
                 if f == folders and future.done() and not future.cancelled()
                 and future.exception() is None]
        if ready:
            _d, i, future = min(ready, key=lambda item: item[:2])
            self.nearest += 1
            return i, future.result()
        self.request(index, folders)
        self.waited += 1
        return key[0], self._jobs[key].result()

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)


def encoder_packet(encoder, library, absolute, source_index, selection, *,
                   background=(4, 4, 4), rotation=0, mirror=False,
                   loop=None, direction=1):
    """One 48 kHz packet of the composited picture, encoded exactly as the
    V7 sender encodes a frame, carrying ``source_index`` and the loop."""
    index, main_folder, float_folder = selection
    value, aspect_code = encoded_picture(
        encoder, library, index, main_folder, float_folder,
        background=background, rotation=rotation, mirror=mirror)
    audio = wire_packet(encoder, value, aspect_code, absolute, source_index,
                        loop=loop, direction=direction)
    return audio, _packet_report(encoder, audio, absolute, source_index,
                                 main_folder, float_folder, aspect_code)


def _packet_report(encoder, audio, absolute, source_index, main_folder,
                   float_folder, aspect_code):
    wire = (encoder.aspect_wire or encoder.mono_wire or encoder.slice_wire)
    return {
        'frame': absolute,
        'source_index': source_index,
        'face_folder': main_folder,
        'float_folder': float_folder,
        'aspect_code': aspect_code,
        'packet_samples': len(audio),
        'peak': float(np.max(np.abs(audio))) if audio.size else 0.0,
        'pcm_clip_samples': int(np.count_nonzero(np.abs(audio) >= 1.0)),
        'limiter_active': False,
        'fold_slots': int(getattr(wire, 'fold_slots', 0) or
                          getattr(encoder.fold, 'slots', 0) or 0),
    }


def _folders(args, library, index, selected, previous):
    if selected is not None:
        return selected, index
    from folder_selector import update_folder_selection, folder_dictionary
    if index != previous:
        update_folder_selection(index, len(library.floats), len(library.mains))
    return folder_dictionary['Main_and_Float_Folders'], index


def _open_midi_input(index_calculator):
    """Open the selected legacy MIDI clock before the live send loop polls it."""
    if not index_calculator.midi_mode:
        return None
    midi = index_calculator.midi_control
    if midi is None or not midi.mido.get_input_names():
        raise RuntimeError('A MIDI clock was selected, but no MIDI input is available')
    midi.midi_control_stuff_main()
    if midi.input_port is None:
        raise RuntimeError('The MIDI clock input could not be opened')
    return midi.input_port


def run_modem(args):
    import settings
    import index_calculator

    source_mode = getattr(args, 'modem_source', 'bake')
    runtime_library = None
    if source_mode == 'images':
        from modem_image_source import RuntimeImageLibrary
        library = RuntimeImageLibrary(
            # Modem mode returns before main.py's shared list-refresh path.
            # Always rebuild so lists from a previous source directory cannot
            # leak into this run.
            rebuild=True,
            capacity=getattr(settings, 'FIFO_LENGTH', 5))
        runtime_library = library
        root = str(library.root)
    else:
        root = args.modem_dir or getattr(settings, 'MODEM_DIR',
                                        settings.IMAGES_DIR + '_modem')
        library = ModemLibrary(root)
    if library.frames > _v7.MAX_SOURCE_INDEX + 1:
        raise ValueError('V7 metadata supports at most 32767 source frames')

    selected = fixed_pair(getattr(args, 'modem_pair', None))
    if selected is not None:
        if selected[0] >= len(library.mains) or selected[1] >= len(library.floats):
            raise ValueError('--modem-pair is outside the selected baked folder lists')
    elif len(library.mains) < 2:
        raise ValueError('The live selector needs a rest folder plus another face folder; '
                         'use --modem-pair 0,0 for one pair')
    if getattr(args, 'modem_wav', None) and selected is None:
        raise ValueError('--modem-wav requires --modem-pair, e.g. 1,0')

    background = getattr(settings, 'BACKGROUND_COLOR', (4, 4, 4))
    rotation = getattr(args, 'rotation', None)
    rotation = rotation if rotation is not None else getattr(settings, 'INITIAL_ROTATION', 0)
    mirror = getattr(args, 'mirror', None)
    mirror = mirror if mirror is not None else bool(getattr(settings, 'INITIAL_MIRROR', 0))
    # The picture encode and wire are the V7 sender's, with its shipped
    # defaults: what an untouched sender GUI sends (tools/v7_send_defaults.py).
    from tools import v7_live
    sender = send_args(args)
    encoder = v7_live.build_send_encoder(sender)
    encoder.warmup()
    profile_name = sender.profile

    channels = pair(args.modem_channels)
    output_latency = latency(args.modem_latency)
    index_calculator.set_clock_mode(args.modem_clock)
    index_calculator.png_paths_len = library.frames
    index_calculator.frame_duration = args.modem_frame_duration
    if index_calculator.midi_mode and args.modem_time_offset_ms:
        raise ValueError('--modem-time-offset-ms requires the free clock; MIDI events cannot be predicted')
    prepare_ms = args.modem_prepare_ms
    receive_margin_ms = args.modem_receive_margin_ms + args.modem_time_offset_ms
    index_offset_ns = round(getattr(args, 'modem_index_offset_ms', 0.0) * 1_000_000)
    # Loop constants for the rotating CRC field: N and the loop phase p (the
    # clock epoch reduced modulo the loop), so any receiver with a correct
    # clock can rebuild the live index and measure how late its picture is.
    # Under a MIDI clock the index does not follow wall time: send N only.
    pingpong = bool(settings.PINGPONG)
    if index_calculator.midi_mode:
        loop = _v7.LoopInfo(library.frames, _v7.LOOP_NO_CLOCK, pingpong)
    else:
        loop = _v7.LoopInfo.from_epoch(library.frames, index_calculator.launch_time,
                                       pingpong)
    previous = None
    sent = 0
    speed = float(getattr(args, 'modem_speed', 1.0))
    if (not math.isfinite(speed) or
            not _v7.MIN_PLAYBACK_SPEED <= speed <= _v7.MAX_PLAYBACK_SPEED):
        raise SystemExit(
            f'[MODEM/V7] --modem-speed must be between '
            f'{_v7.MIN_PLAYBACK_SPEED:g} and {_v7.MAX_PLAYBACK_SPEED:g}')
    print(f'[MODEM/V7] source={source_mode} {root}: {library.frames} images, '
          f'{len(library.mains)} face / {len(library.floats)} float folders; '
          f'wire {_v7.PULSE_FPS*speed:.3f} fps at {speed:g}x, '
          f'source {settings.IPS:g} IPS, '
          f'loop N={loop.frames} p='
          f'{"none (MIDI clock)" if not loop.clocked else loop.phase} in CRC metadata; '
          f'profile={profile_name}; '
          f'encode={"pixel" if sender.pixel_encode else "direct DCT" if sender.dct_encode else "resize"}'
          f'{" linear-light" if sender.dct_encode and not sender.pixel_encode and sender.linear_preshrink else ""}; '
          f'kernel={encoder.kernel_controls.listing()["selected"]}; '
          f'pilot tones=coded')

    def make_packet(absolute, index, folders, at_time_ns=None):
        started = time.perf_counter()
        # Which way the loop is going at this packet's moment: a ping-pong
        # index alone cannot say, and the receiver needs it to compare with
        # its own clock.
        direction = (_v7.loop_direction(_v7.loop_ticks(at_time_ns), loop)
                     if loop.clocked and at_time_ns is not None else 1)
        audio, report = encoder_packet(
            encoder, library, absolute, index, (index, *folders),
            background=background, rotation=rotation, mirror=mirror,
            loop=loop, direction=direction)
        report['direction'] = direction
        report['encode_ms'] = (time.perf_counter() - started) * 1000
        if runtime_library is not None:
            report['fifo_hits'] = runtime_library.fifo_hits
            report['source_index_misses'] = runtime_library.index_misses
        return audio, report

    def prefetch(index, folders, next_index=None):
        if runtime_library is None:
            return
        runtime_library.prefetch(index, *folders)
        if next_index is None:
            next_index = (index + 1) % library.frames
        if library.frames > 1 and next_index != index:
            runtime_library.prefetch(next_index, *folders)

    if getattr(args, 'modem_wav', None):
        path = Path(args.modem_wav)
        path.parent.mkdir(parents=True, exist_ok=True)
        packet_samples = speed_length(_v7.PULSE_FRAME, _v7.RATE, speed)
        # A WAV has no wall clock, so synthesize the same free-clock timeline
        # the live sender samples.  One default pass therefore lasts
        # library.frames / settings.IPS seconds, regardless of packet rate or
        # playback speed.  A requested cycle uses the full folded loop period:
        # 2N ticks for ping-pong, N for a one-way loop.
        limit = _wav_packet_limit(
            library.frames, speed, settings.IPS, pingpong,
            packet_count=args.modem_frames, cycles=args.modem_cycles)
        try:
            with wave.open(str(path), 'wb') as sink:
                sink.setparams((2, 2, _v7.RATE, 0, 'NONE', 'not compressed'))
                folders = selected or (0, 0)
                for n in range(limit):
                    at_time_ns = (index_calculator.launch_time +
                                  (n * packet_samples * 1_000_000_000) // _v7.RATE)
                    index, _ = index_calculator.calculate_free_clock_index(
                        library.frames, pingpong, at_time_ns=at_time_ns,
                        publish=False)
                    next_at_time_ns = (index_calculator.launch_time +
                                       ((n + 1) * packet_samples * 1_000_000_000)
                                       // _v7.RATE)
                    next_index, _ = index_calculator.calculate_free_clock_index(
                        library.frames, pingpong,
                        at_time_ns=next_at_time_ns, publish=False)
                    prefetch(index, folders, next_index)
                    audio, report = make_packet(n + 1, index, folders,
                                                at_time_ns)
                    sink.writeframesraw(pcm(speed_resample(audio, _v7.RATE, speed)))
                    if args.modem_log_frames:
                        print(json.dumps(report), flush=True)
            print(f'[MODEM/V7] Wrote {limit} packets to {path}')
        finally:
            if runtime_library is not None:
                runtime_library.close()
        return

    retry_seconds = 1 / (settings.FPS or 60)
    midi_input = None
    # Pictures are loaded and encoded ahead by workers; the send loop hands
    # a finished one to the wire at its slot.
    ahead = PicturesAhead(
        lambda index, folders: encoded_picture(
            encoder, library, index, *folders, background=background,
            rotation=rotation, mirror=mirror),
        library.frames)
    # Those workers run alongside the audio callback, which needs Python's
    # lock within one output buffer (2.7 ms at low latency). Python hands the
    # lock over every 5 ms by default; hand it over sooner.
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(min(switch_interval, .0005))
    # Have the first packets' pictures finished before the sound starts: a
    # first packet that waits for its picture is late, and the output skips.
    now_ns = time.time_ns()
    packet_seconds = _v7.PULSE_FRAME/_v7.RATE/speed
    first, _ = index_calculator.calculate_free_clock_index(
        library.frames, settings.PINGPONG, at_time_ns=now_ns,
        time_offset_ns=index_offset_ns, publish=False)
    first = max(0, min(int(first), library.frames - 1))
    first_folders, previous = _folders(args, library, first, selected, previous)
    priming = []
    for step in range(AHEAD_PACKETS+1):
        later, _ = index_calculator.calculate_free_clock_index(
            library.frames, settings.PINGPONG,
            at_time_ns=now_ns + round(step*packet_seconds*1e9),
            time_offset_ns=index_offset_ns, publish=False)
        priming.append(max(0, min(int(later), library.frames - 1)))
    for wanted in priming:
        ahead.request(wanted, first_folders)
    for wanted in priming:
        ahead.take(wanted, first_folders)        # waits for each to finish
    try:
        midi_input = _open_midi_input(index_calculator)
        with PacketOutput(device(args.scope_device), channels, output_latency,
                          frame=_v7.PULSE_FRAME, packet=_v7.PULSE_FRAME,
                          speed=speed,
                          block=getattr(args, 'modem_block_samples', None)) as output:
            print(f'[MODEM/V7] output {output.rate:g} Hz, {output.fps:.2f} fps')
            packet_ns = round(output.emit_frame/output.rate * 1_000_000_000)
            while not args.modem_frames or sent < args.modem_frames:
                started = time.perf_counter()
                output.wait_ready()
                index, _ = index_calculator.update_index(library.frames, settings.PINGPONG)
                index = max(0, min(int(index), library.frames - 1))
                slot = None
                target_time_ns = None
                next_index = None
                if not index_calculator.midi_mode:
                    slot = output.reserve(prepare_ms, receive_margin_ms)
                    if slot is not None:
                        target_time_ns = slot.target_time_ns
                        index, _ = index_calculator.calculate_free_clock_index(
                            library.frames, settings.PINGPONG,
                            at_time_ns=target_time_ns,
                            time_offset_ns=index_offset_ns, publish=False)
                        index = max(0, min(int(index), library.frames - 1))
                        next_at_time_ns = (
                            target_time_ns +
                            round(output.emit_frame/output.rate * 1_000_000_000))
                        next_index, _ = index_calculator.calculate_free_clock_index(
                            library.frames, settings.PINGPONG,
                            at_time_ns=next_at_time_ns,
                            time_offset_ns=index_offset_ns, publish=False)
                        next_index = max(0, min(int(next_index),
                                                library.frames - 1))
                folders, previous = _folders(args, library, index, selected, previous)
                prefetch(index, folders, next_index)
                # Ask for this packet's picture and the next few packets'
                # (by the clock at their send times), then take the finished
                # one for this slot.
                upcoming = [index]
                for step in range(1, AHEAD_PACKETS+1):
                    if target_time_ns is None:
                        upcoming.append((index+step) % library.frames)
                        continue
                    later, _ = index_calculator.calculate_free_clock_index(
                        library.frames, settings.PINGPONG,
                        at_time_ns=target_time_ns + step*packet_ns,
                        time_offset_ns=index_offset_ns, publish=False)
                    upcoming.append(max(0, min(int(later), library.frames - 1)))
                for wanted in upcoming:
                    ahead.request(wanted, folders)
                handoff = time.perf_counter()
                sent_index, (value, aspect_code) = ahead.take(index, folders)
                direction = (_v7.loop_direction(_v7.loop_ticks(target_time_ns), loop)
                             if loop.clocked and target_time_ns is not None else 1)
                audio = wire_packet(encoder, value, aspect_code, sent + 1,
                                    sent_index, loop=loop, direction=direction)
                report = _packet_report(encoder, audio, sent + 1, sent_index,
                                        *folders, aspect_code)
                report.update(direction=direction, wanted_index=index,
                              encode_ms=(time.perf_counter()-handoff)*1000)
                if runtime_library is not None:
                    report['fifo_hits'] = runtime_library.fifo_hits
                    report['source_index_misses'] = runtime_library.index_misses
                prepare_ms = max(args.modem_prepare_ms, report['encode_ms'] * 1.5)
                if output.submit(audio, slot):
                    sent += 1
                    if args.modem_log_frames:
                        report['target_time_ns'] = target_time_ns
                        print(json.dumps(report), flush=True)
                else:
                    # A missed reservation leaves the output ready. Yield before
                    # recomputing/re-encoding rather than retrying in a tight loop.
                    delay = retry_seconds - (time.perf_counter() - started)
                    if delay > 0:
                        time.sleep(delay)
            output.finish()
    finally:
        ahead.close()
        sys.setswitchinterval(switch_interval)
        try:
            if midi_input is not None:
                midi_input.close()
        finally:
            if runtime_library is not None:
                runtime_library.close()
