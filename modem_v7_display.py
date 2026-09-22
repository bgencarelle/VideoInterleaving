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
from animation_modem.v7_core import speed_resample
from animation_modem.playback import PacketOutput, latency
from modem_bake import ModemLibrary
from animation_modem import v7 as _v7


TARGET_RMS = .1521


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
           encode_filter='nearest'):
    index, main_folder, float_folder = selection
    image = library.composite(index, main_folder, float_folder,
                              background, rotation, mirror)
    values = _source_values(model, image, encode_filter)
    audio = _v7.encode_pulse_frame(
        model, values, absolute,
        aspect_code=_v7.aspect_wire_code(image.size),
        source_index=source_index)
    return audio, {
        'frame': absolute,
        'source_index': source_index,
        'face_folder': main_folder,
        'float_folder': float_folder,
        'packet_samples': len(audio),
    }


def _folders(args, library, index, selected, previous):
    if selected is not None:
        return selected, index
    from folder_selector import update_folder_selection, folder_dictionary
    if index != previous:
        update_folder_selection(index, len(library.floats), len(library.mains))
    return folder_dictionary['Main_and_Float_Folders'], index


def run_modem(args):
    import settings
    import index_calculator

    source_mode = getattr(args, 'modem_source', 'bake')
    runtime_library = None
    if source_mode == 'images':
        from modem_image_source import RuntimeImageLibrary
        library = RuntimeImageLibrary(
            rebuild=bool(getattr(args, 'rebuild', False)),
            capacity=getattr(settings, 'FIFO_LENGTH', 5))
        runtime_library = library
        root = str(library.root)
    else:
        root = args.modem_dir or getattr(settings, 'MODEM_DIR',
                                        settings.IMAGES_DIR + '_modem')
        library = ModemLibrary(root)
    if library.frames > 0xffff:
        raise ValueError('V7 source index supports at most 65535 source frames')

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
    previous = None
    sent = 0
    speed = float(getattr(args, 'modem_speed', 1.0))
    print(f'[MODEM/V7] source={source_mode} {root}: {library.frames} images, '
          f'{len(library.mains)} face / {len(library.floats)} float folders; '
          f'wire {_v7.PULSE_FPS*speed:.3f} fps at {speed:g}x, '
          'source index in CRC metadata')

    def make_packet(absolute, index, folders):
        started = time.perf_counter()
        audio, report = packet(
            library, model, absolute, index, (index, *folders),
            background=background, rotation=rotation, mirror=mirror,
            encode_filter=encode_filter)
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
        limit = args.modem_frames or library.frames
        try:
            with wave.open(str(path), 'wb') as sink:
                sink.setparams((2, 2, _v7.RATE, 0, 'NONE', 'not compressed'))
                folders = selected or (0, 0)
                for n in range(limit):
                    index = n % library.frames
                    prefetch(index, folders)
                    audio, report = make_packet(n + 1, index, folders)
                    sink.writeframesraw(pcm(speed_resample(audio, _v7.RATE, speed)))
                    if args.modem_log_frames:
                        print(json.dumps(report), flush=True)
            print(f'[MODEM/V7] Wrote {limit} packets to {path}')
        finally:
            if runtime_library is not None:
                runtime_library.close()
        return

    poll_seconds = 1 / (settings.FPS or 60)
    try:
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
                    audio, report = make_packet(sent + 1, index, folders)
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
        if runtime_library is not None:
            runtime_library.close()
