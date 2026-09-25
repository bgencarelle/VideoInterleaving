"""V7 pulse-wire sender using the application's shared image clock.

This is the application adapter around the standalone V7 transport.  The V7
encoder remains in ``tools.v7_proto`` so the standalone bench and live sender
use the same implementation; this module owns source selection, shared-clock
scheduling, and audio output.
"""
import json
import math
from pathlib import Path
import time
import wave

import numpy as np

from animation_modem.audio_common import device, pair, pcm
from animation_modem.v7_core import speed_length, speed_resample
from animation_modem.playback import PacketOutput, latency
from modem_bake import ModemLibrary
from animation_modem import v7 as _v7


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


def packet(library, model, absolute, source_index, selection, *,
           background=(4, 4, 4), rotation=0, mirror=False,
           encode_filter='nearest', loop=None, direction=1,
           pilot_tones=True, eof_marker=True):
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
    audio = _v7.encode_pulse_frame(
        model, values, absolute,
        aspect_code=aspect_code,
        source_index=source_index, loop=loop, direction=direction,
        pilot_tones=pilot_tones, eof_marker=eof_marker)
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
    encode_filter = getattr(args, 'modem_encode_filter', None) or 'nearest'
    pilot_tones = bool(getattr(args, 'modem_pilot_tones', True))
    eof_marker = bool(getattr(args, 'modem_eof_marker', True))

    # The V7 statistics are a wire profile, not a property of whichever frame
    # happens to be sent first.  The canonical profile ships as frozen,
    # hash-checked tables (animation_modem/v7_model_tables.npz), so the
    # application sender and the standalone receiver build identical models
    # without the reference image or any runtime derivation.
    model = _v7.load_model(
        TARGET_RMS / math.sqrt(1 + 10**(_v7.CLOCK_REL_DB/10)),
        encode_filter=encode_filter)

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
          f'EOF marker={"on" if eof_marker else "off"}; '
          f'pilot tones={"on" if pilot_tones else "off"}')

    def make_packet(absolute, index, folders, at_time_ns=None):
        started = time.perf_counter()
        # Which way the loop is going at this packet's moment: a ping-pong
        # index alone cannot say, and the receiver needs it to compare with
        # its own clock.
        direction = (_v7.loop_direction(_v7.loop_ticks(at_time_ns), loop)
                     if loop.clocked and at_time_ns is not None else 1)
        audio, report = packet(
            library, model, absolute, index, (index, *folders),
            background=background, rotation=rotation, mirror=mirror,
            encode_filter=encode_filter, loop=loop, direction=direction,
            pilot_tones=pilot_tones, eof_marker=eof_marker)
        report['direction'] = direction
        report['encode_ms'] = (time.perf_counter() - started) * 1000
        if runtime_library is not None:
            report['fifo_hits'] = runtime_library.fifo_hits
            report['source_index_misses'] = runtime_library.index_misses
        return audio, report

    def prefetch(index, folders):
        if runtime_library is None:
            return
        runtime_library.prefetch(index, *folders)
        if library.frames > 1:
            runtime_library.prefetch((index + 1) % library.frames, *folders)

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
                    prefetch(index, folders)
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

    poll_seconds = 1 / (settings.FPS or 60)
    midi_input = None
    try:
        midi_input = _open_midi_input(index_calculator)
        with PacketOutput(device(args.scope_device), channels, output_latency,
                          frame=_v7.PULSE_FRAME, packet=_v7.PULSE_FRAME,
                          speed=speed) as output:
            print(f'[MODEM/V7] output {output.rate:g} Hz, {output.fps:.2f} fps')
            while not args.modem_frames or sent < args.modem_frames:
                started = time.perf_counter()
                index, _ = index_calculator.update_index(library.frames, settings.PINGPONG)
                index = max(0, min(int(index), library.frames - 1))
                if output.ready():
                    slot = None
                    target_time_ns = None
                    if not index_calculator.midi_mode:
                        slot = output.reserve(prepare_ms, receive_margin_ms)
                        if slot is not None:
                            target_time_ns = slot.target_time_ns
                            index, _ = index_calculator.calculate_free_clock_index(
                                library.frames, settings.PINGPONG,
                                at_time_ns=target_time_ns,
                                time_offset_ns=index_offset_ns, publish=False)
                            index = max(0, min(int(index), library.frames - 1))
                    folders, previous = _folders(args, library, index, selected, previous)
                    prefetch(index, folders)
                    audio, report = make_packet(sent + 1, index, folders,
                                                target_time_ns)
                    prepare_ms = max(args.modem_prepare_ms, report['encode_ms'] * 1.5)
                    if output.submit(audio, slot):
                        sent += 1
                        if args.modem_log_frames:
                            report['target_time_ns'] = target_time_ns
                            print(json.dumps(report), flush=True)
                delay = poll_seconds - (time.perf_counter() - started)
                if delay > 0:
                    time.sleep(delay)
            output.finish()
    finally:
        try:
            if midi_input is not None:
                midi_input.close()
        finally:
            if runtime_library is not None:
                runtime_library.close()
