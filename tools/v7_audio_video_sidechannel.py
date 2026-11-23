#!/usr/bin/env python3
"""Carry a V7 picture-detail layer through a listenable source-audio leg.

This bench uses an audiovisual fixture, puts the ordinary mono V7 picture on
the right leg, and phase-QIMs a real luma-residual bitplane into active blocks
of the soundtrack on the left leg. The receiver derives active audio blocks
from energy, decodes the payload, checks one CRC-16 per picture, and applies the
recovered residual map to the V7 reconstruction.

The pulse sweep remains as a clean modulation reference. Media results include
the AAC soundtrack's waveform impact and the base/enhanced picture scores. The
audio embedding is a digital-channel experiment; its level and bit rate need
listening and channel validation before they can be treated as operating
settings.
"""
import argparse
import binascii
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from animation_modem import v7
from animation_modem.imaging import values_image
from tools.v7_torture_matrix import TARGET


DEFAULT_FIXTURE = ROOT/'modem_tests/fixtures/v7_robot_count_sync_test.mp4'
SWEEP_LOW_HZ = 300.0
SWEEP_HIGH_HZ = 12_750.0
SWEEP_PEAK = .18
SWEEP_PULSE_MS = 20.0
SWEEP_PERIOD_MS = 25.0
BLOCK_SAMPLES = 960                 # 20 ms at 48 kHz
BLOCK_HOP_SAMPLES = 1200            # 25 ms at 48 kHz
ACTIVE_AUDIO_RMS = .001
PHASE_COSET_PERIODS = 64
DEFAULT_BINS_PER_BLOCK = 160
DEFAULT_BITS_PER_BIN = 4
DETAIL_GRID = (32, 40)              # 1-bit residual map, rows x columns
DETAIL_CORRECTION = 8               # luma codes applied at the receiver
CRC_BYTES = 2


def pulse_sweep_spec(sample_count, sample_rate=v7.RATE,
                     low_hz=SWEEP_LOW_HZ, high_hz=SWEEP_HIGH_HZ,
                     pulse_ms=SWEEP_PULSE_MS,
                     period_ms=SWEEP_PERIOD_MS):
    """Return complete pulse positions and the stepped reference frequencies."""
    sample_count = int(sample_count)
    sample_rate = int(sample_rate)
    low_hz, high_hz = float(low_hz), float(high_hz)
    pulse_ms, period_ms = float(pulse_ms), float(period_ms)
    if sample_count < 0 or sample_rate <= 0:
        raise ValueError('sample count must be non-negative and rate positive')
    if not 0 < low_hz < high_hz < sample_rate/2:
        raise ValueError('sweep band must be ordered, positive, and below Nyquist')
    if not (np.isfinite(pulse_ms) and np.isfinite(period_ms) and
            0 < pulse_ms <= period_ms):
        raise ValueError('pulse and period must be finite, with 0 < pulse <= period')
    pulse_samples = max(2, round(sample_rate*pulse_ms/1000))
    period_samples = max(pulse_samples, round(sample_rate*period_ms/1000))
    if sample_count < pulse_samples:
        return pulse_samples, period_samples, np.empty(0, dtype=int), np.empty(0)
    count = 1+(sample_count-pulse_samples)//period_samples
    starts = np.arange(count, dtype=int)*period_samples
    frequencies = np.linspace(low_hz, high_hz, count)
    return pulse_samples, period_samples, starts, frequencies


def pulse_sweep_reference(sample_count, sample_rate=v7.RATE,
                          low_hz=SWEEP_LOW_HZ,
                          high_hz=SWEEP_HIGH_HZ, peak=SWEEP_PEAK,
                          pulse_ms=SWEEP_PULSE_MS,
                          period_ms=SWEEP_PERIOD_MS):
    """Create complete Hann-windowed reference pulses; omit a truncated pulse."""
    peak = float(peak)
    if not np.isfinite(peak) or not 0 <= peak <= 1:
        raise ValueError('pulse peak must be between 0 and 1')
    pulse_samples, _, starts, frequencies = pulse_sweep_spec(
        sample_count, sample_rate, low_hz, high_hz, pulse_ms, period_ms)
    time = np.arange(pulse_samples, dtype=np.float64)/int(sample_rate)
    window = np.hanning(pulse_samples+2)[1:-1]
    audio = np.zeros(int(sample_count), dtype=np.float32)
    for start, frequency in zip(starts, frequencies):
        tone = np.sin(2*np.pi*frequency*time)
        audio[start:start+pulse_samples] = (peak*tone*window).astype(np.float32)
    return audio


def _phase_qim_target(phase, symbol, bits_per_component,
                      coset_periods=PHASE_COSET_PERIODS):
    bits_per_component, coset_periods = int(bits_per_component), int(coset_periods)
    if not 1 <= bits_per_component <= 8 or coset_periods <= 0:
        raise ValueError('phase QIM supports 1..8 bits and positive coset count')
    levels = 1 << bits_per_component
    if not 0 <= int(symbol) < levels:
        raise ValueError('phase symbol is outside its constellation')
    step = 2*np.pi/(coset_periods*levels)
    period = levels*step
    index = int(np.rint((float(phase)-int(symbol)*step)/period))
    return (index*period+int(symbol)*step) % (2*np.pi)


def _phase_qim_decode(phase, bits_per_component,
                      coset_periods=PHASE_COSET_PERIODS):
    bits_per_component, coset_periods = int(bits_per_component), int(coset_periods)
    if not 1 <= bits_per_component <= 8 or coset_periods <= 0:
        raise ValueError('phase QIM supports 1..8 bits and positive coset count')
    levels = 1 << bits_per_component
    step = 2*np.pi/(coset_periods*levels)
    return int(np.rint((float(phase) % (2*np.pi))/step)) % levels


def _select_bins(spectrum, sample_rate, low_hz, high_hz, count):
    n = (len(spectrum)-1)*2
    frequencies = np.fft.rfftfreq(n, 1/sample_rate)
    candidates = np.flatnonzero((frequencies >= low_hz) &
                                (frequencies <= high_hz))
    if not len(candidates):
        return np.empty(0, dtype=int)
    count = min(int(count), len(candidates))
    order = np.argsort(np.abs(spectrum[candidates]), kind='stable')[-count:]
    return np.sort(candidates[order])


def active_audio_starts(audio, block_samples=BLOCK_SAMPLES,
                        hop_samples=BLOCK_HOP_SAMPLES,
                        rms_threshold=ACTIVE_AUDIO_RMS):
    """Find complete blocks with enough host energy for phase decoding."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    block_samples, hop_samples = int(block_samples), int(hop_samples)
    rms_threshold = float(rms_threshold)
    if block_samples <= 0 or hop_samples <= 0:
        raise ValueError('audio block and hop must be positive')
    if not np.isfinite(rms_threshold) or rms_threshold < 0:
        raise ValueError('audio activity threshold must be finite and non-negative')
    starts = np.arange(0, max(0, len(audio)-block_samples+1), hop_samples,
                       dtype=int)
    active = []
    for start in starts:
        block = audio[start:start+block_samples].astype(np.float64)
        if np.sqrt(np.mean(block*block)) > rms_threshold:
            active.append(start)
    return np.asarray(active, dtype=int)


def encode_phase_qim(audio, starts, block_samples, bins_per_block,
                     bits_per_component, coset_periods=PHASE_COSET_PERIODS,
                     payload_bits=None, sample_rate=v7.RATE,
                     low_hz=SWEEP_LOW_HZ, high_hz=SWEEP_HIGH_HZ,
                     seed=0xA0D10):
    """Encode supplied bits in audio FFT phases while preserving magnitudes."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    starts = np.asarray(starts, dtype=int)
    block_samples = int(block_samples)
    bins_per_block = int(bins_per_block)
    bits_per_component = int(bits_per_component)
    coset_periods = int(coset_periods)
    if not len(starts) or bins_per_block <= 0:
        raise ValueError('at least one block and a positive bin count are required')
    if not 1 <= bits_per_component <= 8 or coset_periods <= 0:
        raise ValueError('invalid phase-QIM constellation')
    capacity = len(starts)*bins_per_block*bits_per_component
    rng = np.random.default_rng(seed)
    if payload_bits is None:
        payload = rng.integers(0, 2, capacity, dtype=np.uint8)
    else:
        payload = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
        if np.any(payload > 1):
            raise ValueError('payload must contain binary bits')
        if len(payload) > capacity:
            raise ValueError(f'payload needs {len(payload)} bits, capacity is {capacity}')
    padding = rng.integers(0, 2, capacity-len(payload), dtype=np.uint8)
    carrier_bits = np.concatenate((payload, padding))
    weights = 1 << np.arange(bits_per_component-1, -1, -1)
    symbols = np.sum(carrier_bits.reshape(-1, bits_per_component)*weights,
                     axis=1).reshape(len(starts), bins_per_block)
    output = audio.copy()
    selected_bins = np.empty((len(starts), bins_per_block), dtype=int)
    max_phase_shift = 0.0

    for block_index, start in enumerate(starts):
        stop = int(start)+block_samples
        if start < 0 or stop > len(output):
            raise ValueError('audio block schedule extends beyond the samples')
        spectrum = np.fft.rfft(output[start:stop].astype(np.float64))
        bins = _select_bins(spectrum, sample_rate, low_hz, high_hz,
                            bins_per_block)
        if len(bins) != bins_per_block:
            raise ValueError('not enough audio bins for the requested payload')
        selected_bins[block_index] = bins
        for slot, bin_index in enumerate(bins):
            phase = float(np.angle(spectrum[bin_index]) % (2*np.pi))
            target = _phase_qim_target(
                phase, symbols[block_index, slot], bits_per_component,
                coset_periods)
            delta = (target-phase+np.pi) % (2*np.pi)-np.pi
            max_phase_shift = max(max_phase_shift, abs(float(delta)))
            spectrum[bin_index] = abs(spectrum[bin_index])*np.exp(1j*target)
        output[start:stop] = np.fft.irfft(
            spectrum, n=block_samples).astype(np.float32)

    return {
        'audio': output,
        'payload_bits': payload.copy(),
        'carrier_capacity_bits': capacity,
        'selected_bins': selected_bins,
        'max_phase_shift_rad': max_phase_shift,
    }


def decode_phase_qim(audio, starts, block_samples, bins_per_block,
                     bits_per_component, coset_periods=PHASE_COSET_PERIODS,
                     sample_rate=v7.RATE, low_hz=SWEEP_LOW_HZ,
                     high_hz=SWEEP_HIGH_HZ):
    """Recover bits by selecting the strongest bins from received magnitudes."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    starts = np.asarray(starts, dtype=int)
    values = []
    for start in starts:
        stop = int(start)+int(block_samples)
        if start < 0 or stop > len(audio):
            raise ValueError('received audio block schedule is out of range')
        spectrum = np.fft.rfft(audio[start:stop].astype(np.float64))
        bins = _select_bins(spectrum, sample_rate, low_hz, high_hz,
                            bins_per_block)
        if len(bins) != int(bins_per_block):
            raise ValueError('not enough received audio bins for the allocation')
        values.extend(_phase_qim_decode(
            np.angle(spectrum[index]), bits_per_component, coset_periods)
                      for index in bins)
    symbols = np.asarray(values, dtype=np.uint16)
    shifts = np.arange(int(bits_per_component)-1, -1, -1)
    return ((symbols[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def plan_payload_schedule(active_starts, frame_count, bits_per_frame,
                          packet_samples=v7.PULSE_FRAME,
                          block_capacity=DEFAULT_BINS_PER_BLOCK*
                          DEFAULT_BITS_PER_BIN,
                          block_samples=BLOCK_SAMPLES,
                          sample_rate=v7.RATE):
    """Schedule a frame-bit FIFO only into audio blocks after frame release."""
    active_starts = np.asarray(active_starts, dtype=int)
    frame_count, bits_per_frame = int(frame_count), int(bits_per_frame)
    packet_samples, block_capacity = int(packet_samples), int(block_capacity)
    if frame_count <= 0 or bits_per_frame <= 0 or block_capacity <= 0:
        raise ValueError('frame count and payload capacities must be positive')
    used_starts, block_bit_counts = [], []
    frame_ready = np.full(frame_count, np.nan, dtype=np.float64)
    released_frames = sent_bits = completed_frames = 0
    total_bits = frame_count*bits_per_frame

    for start in active_starts:
        while (released_frames < frame_count and
               released_frames*packet_samples <= int(start)):
            released_frames += 1
        available = released_frames*bits_per_frame-sent_bits
        if available <= 0:
            continue
        count = min(block_capacity, available)
        sent_bits += count
        used_starts.append(int(start))
        block_bit_counts.append(count)
        while (completed_frames < frame_count and
               (completed_frames+1)*bits_per_frame <= sent_bits):
            frame_ready[completed_frames] = (
                int(start)+int(block_samples))/float(sample_rate)
            completed_frames += 1
        if sent_bits == total_bits:
            break

    if sent_bits < total_bits:
        raise ValueError(
            f'audio schedule carries {sent_bits}/{total_bits} payload bits; '
            'extend the clip or lower the detail payload rate')
    return {
        'starts': np.asarray(used_starts, dtype=int),
        'block_bit_counts': np.asarray(block_bit_counts, dtype=int),
        'frame_ready_seconds': frame_ready,
        'payload_bits': sent_bits,
    }


def make_detail_payload(source_luma, base_image, grid=DETAIL_GRID):
    """Make a CRC-protected sign map of the high-resolution luma residual."""
    source_luma = np.asarray(source_luma, dtype=np.uint8)
    rows, columns = (int(grid[0]), int(grid[1]))
    if source_luma.ndim != 2 or rows <= 0 or columns <= 0:
        raise ValueError('source luma and detail grid must be valid')
    height, width = source_luma.shape
    if rows > height or columns > width:
        raise ValueError('detail grid cannot exceed source image dimensions')
    base_luma = np.asarray(
        base_image.resize((width, height), Image.Resampling.BICUBIC)
        .convert('L'), dtype=np.float32)
    residual = source_luma.astype(np.float32)-base_luma
    bits = []
    for ys in np.array_split(np.arange(height), rows):
        for xs in np.array_split(np.arange(width), columns):
            bits.append(float(np.mean(residual[np.ix_(ys, xs)])) >= 0)
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), bitorder='big').tobytes()
    crc = binascii.crc_hqx(packed, 0xFFFF)
    return packed+crc.to_bytes(CRC_BYTES, 'big')


def unpack_detail_payload(payload, grid=DETAIL_GRID):
    """Return the residual bitplane when its frame CRC-16 is valid."""
    rows, columns = (int(grid[0]), int(grid[1]))
    data_bytes = (rows*columns+7)//8
    payload = bytes(payload)
    if len(payload) != data_bytes+CRC_BYTES:
        return None
    data = payload[:data_bytes]
    expected = int.from_bytes(payload[data_bytes:], 'big')
    if binascii.crc_hqx(data, 0xFFFF) != expected:
        return None
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8),
                         bitorder='big')[:rows*columns].reshape(rows, columns)


def apply_detail_payload(base_image, residual_bits, output_size,
                         correction=DETAIL_CORRECTION):
    """Apply the decoded sign map as a small luma residual correction."""
    residual_bits = np.asarray(residual_bits, dtype=np.uint8)
    width, height = (int(output_size[0]), int(output_size[1]))
    if residual_bits.ndim != 2 or min(width, height) <= 0:
        raise ValueError('residual map and output size must be valid')
    cell_values = (2*residual_bits.astype(np.float32)-1)*float(correction)
    correction_image = Image.fromarray(cell_values, mode='F').resize(
        (width, height), Image.Resampling.NEAREST)
    luma_correction = np.asarray(correction_image, dtype=np.float32)
    ycbcr = np.asarray(base_image.resize(
        (width, height), Image.Resampling.BICUBIC).convert('YCbCr'))
    ycbcr = ycbcr.copy()
    ycbcr[:, :, 0] = np.clip(
        ycbcr[:, :, 0].astype(np.float32)+luma_correction, 0, 255).astype(
            np.uint8)
    return Image.fromarray(ycbcr, mode='YCbCr').convert('RGB')


def _extract_audio(path, sample_rate=v7.RATE):
    command = [
        'ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0', '-vn',
        '-ac', '1', '-ar', str(int(sample_rate)), '-f', 'f32le', 'pipe:1',
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        message = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(f'could not decode fixture audio: {message}')
    if not result.stdout:
        raise ValueError('media fixture has no audio stream')
    return np.frombuffer(result.stdout, dtype='<f4').astype(np.float32, copy=True)


def _read_video_frames(path, audio_samples, frame_limit=None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'could not open video fixture: {path}')
    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(source_fps) or source_fps <= 0 or source_count <= 0:
            raise ValueError('fixture video has invalid rate or frame count')
        count = min(int(source_count*v7.PULSE_FPS/source_fps),
                    int(audio_samples)//v7.PULSE_FRAME)
        if frame_limit is not None:
            count = min(count, int(frame_limit))
        target_indices = np.rint(
            np.arange(count)*source_fps/v7.PULSE_FPS).astype(int)
        while count and target_indices[-1] >= source_count:
            count -= 1
            target_indices = target_indices[:count]
        if count < 3:
            raise ValueError('at least three paired A/V frames are required')

        prepared, source_luma = [], []
        source_index = -1
        bgr = None
        for target in target_indices:
            while source_index < target:
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError('video ended before its reported frame count')
                source_index += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            source_luma.append(np.asarray(image.convert('L'), dtype=np.uint8))
            prepared.append(v7.prepare_image(image, 'box'))
        return source_fps, prepared, source_luma
    finally:
        capture.release()


def _decode_picture_leg(wire, mono_model, video_leg, frame_count):
    with wire.receiving():
        results, info = v7.decode_pulse_stream(
            mono_model, np.asarray(video_leg, dtype=np.float32)[:, None],
            sample_rate=v7.RATE, pilot_timing='tone-seeded', state=v7.PulseState(tail_memory=False))
    images = [None]*frame_count
    for result in results:
        if not result.diag.get('displayable'):
            continue
        index = int(result.counter)-1
        if 0 <= index < frame_count:
            images[index] = values_image(
                wire.values(mono_model, result), mono_model.coder.grids)
    if any(image is None for image in images):
        raise RuntimeError('the base V7 video leg did not decode every frame')
    return images, info


def _psnr(reference, candidate):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    mse = float(np.mean((a-b)**2))
    return float('inf') if mse == 0 else 10*np.log10(255**2/mse)


def _audio_metrics(reference, modified, modified_starts=None,
                   block_samples=BLOCK_SAMPLES):
    reference = np.asarray(reference, dtype=np.float64)
    modified = np.asarray(modified, dtype=np.float64)
    error = modified-reference
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    err_rms = float(np.sqrt(np.mean(error**2)))
    metrics = {
        'source_rms': ref_rms,
        'modification_rms': err_rms,
        'waveform_snr_db': (float(20*np.log10(ref_rms/err_rms))
                            if err_rms else float('inf')),
        'waveform_correlation': float(np.corrcoef(reference, modified)[0, 1]),
        'peak': float(np.max(np.abs(modified))),
        'clipped_samples': int(np.count_nonzero(np.abs(modified) >= 1.0)),
    }
    if modified_starts is not None:
        active = np.zeros(len(reference), dtype=bool)
        for start in np.asarray(modified_starts, dtype=int):
            active[start:start+int(block_samples)] = True
        active_error_rms = float(np.sqrt(np.mean(error[active]**2)))
        active_source_rms = float(np.sqrt(np.mean(reference[active]**2)))
        metrics['modified_blocks_snr_db'] = (
            float(20*np.log10(active_source_rms/active_error_rms))
            if active_error_rms else float('inf'))
    return metrics


def _sweep_reference_metrics():
    samples = 12*v7.PULSE_FRAME
    audio = pulse_sweep_reference(samples)
    block_samples, _, starts, _ = pulse_sweep_spec(samples)
    encoded = encode_phase_qim(
        audio, starts, block_samples, bins_per_block=16,
        bits_per_component=4, coset_periods=PHASE_COSET_PERIODS)
    decoded = decode_phase_qim(
        encoded['audio'], starts, block_samples, bins_per_block=16,
        bits_per_component=4, coset_periods=PHASE_COSET_PERIODS)
    errors = int(np.count_nonzero(decoded != encoded['payload_bits']))
    metrics = _audio_metrics(audio, encoded['audio'], starts, block_samples)
    return {
        'kind': '20 ms Hann pulses every 25 ms, 300 Hz to 12.75 kHz',
        'pulse_count': len(starts),
        'payload_bits': int(len(encoded['payload_bits'])),
        'payload_bps': len(encoded['payload_bits'])/(samples/v7.RATE),
        'bit_errors': errors,
        'ber': errors/len(encoded['payload_bits']),
        **metrics,
    }


def run(out, fixture=DEFAULT_FIXTURE, frames=None,
        activity_rms=ACTIVE_AUDIO_RMS,
        bins_per_block=DEFAULT_BINS_PER_BLOCK,
        bits_per_component=DEFAULT_BITS_PER_BIN,
        coset_periods=PHASE_COSET_PERIODS,
        detail_grid=DETAIL_GRID,
        correction=DETAIL_CORRECTION):
    fixture = Path(fixture)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bins_per_block = int(bins_per_block)
    bits_per_component = int(bits_per_component)
    coset_periods = int(coset_periods)
    detail_grid = tuple(int(value) for value in detail_grid)
    if frames is not None and int(frames) < 3:
        raise ValueError('at least three V7 packets are required for EOF framing')
    if bins_per_block <= 0 or not 1 <= bits_per_component <= 8:
        raise ValueError('audio bin and bit allocations must be positive')
    if coset_periods <= 0 or len(detail_grid) != 2 or min(detail_grid) <= 0:
        raise ValueError('phase cosets and detail grid must be positive')

    audio_full = _extract_audio(fixture)
    source_fps, prepared, source_luma = _read_video_frames(
        fixture, len(audio_full), frames)
    frame_count = len(prepared)
    sample_count = frame_count*v7.PULSE_FRAME
    audio = audio_full[:sample_count].copy()

    from tools.v7_live import _ensure_test_modem_path
    _ensure_test_modem_path()
    from mono_video import MonoFreshFoldWire

    model = v7.load_model(TARGET, 'box')
    values = [v7.image_values(image, model.coder.grids, 'box')
              for image in prepared]
    wire = MonoFreshFoldWire(model, side='right')
    mono_model = wire.model_for(model)
    encoded_stereo = wire.encode(
        model, values, source_indices=list(range(frame_count)))
    video_leg = np.asarray(encoded_stereo[:, 1], dtype=np.float32)
    base_images, video_info = _decode_picture_leg(
        wire, mono_model, video_leg, frame_count)

    payloads = [make_detail_payload(luma, base, detail_grid)
                for luma, base in zip(source_luma, base_images)]
    bits_per_frame = len(payloads[0])*8
    if any(len(payload) != len(payloads[0]) for payload in payloads):
        raise RuntimeError('detail frame payload sizes are inconsistent')
    payload_bits = np.unpackbits(np.frombuffer(b''.join(payloads),
                                              dtype=np.uint8))

    active_starts = active_audio_starts(
        audio, rms_threshold=activity_rms)
    bits_per_block = bins_per_block*bits_per_component
    schedule = plan_payload_schedule(
        active_starts, frame_count, bits_per_frame,
        block_capacity=bits_per_block)
    used_starts = schedule['starts']
    encoded = encode_phase_qim(
        audio, used_starts, BLOCK_SAMPLES, bins_per_block,
        bits_per_component, coset_periods, payload_bits=payload_bits)

    receiver_active = active_audio_starts(
        encoded['audio'], rms_threshold=activity_rms)
    receive_schedule = plan_payload_schedule(
        receiver_active, frame_count, bits_per_frame,
        block_capacity=bits_per_block)
    if not np.array_equal(used_starts, receive_schedule['starts']):
        raise RuntimeError('receiver activity gate disagrees with the sender')
    decoded_carrier = decode_phase_qim(
        encoded['audio'], receive_schedule['starts'], BLOCK_SAMPLES,
        bins_per_block, bits_per_component, coset_periods)
    decoded_bits = decoded_carrier[:len(payload_bits)]
    bit_errors = int(np.count_nonzero(decoded_bits != payload_bits))

    enhanced_images = []
    crc_valid = 0
    for index, (base, source) in enumerate(zip(base_images, source_luma)):
        start = index*bits_per_frame
        end = start+bits_per_frame
        frame_bytes = np.packbits(decoded_bits[start:end]).tobytes()
        residual_bits = unpack_detail_payload(frame_bytes, detail_grid)
        if residual_bits is None:
            enhanced_images.append(base.resize(
                (source.shape[1], source.shape[0]), Image.Resampling.BICUBIC))
            continue
        crc_valid += 1
        enhanced_images.append(apply_detail_payload(
            base, residual_bits, (source.shape[1], source.shape[0]), correction))

    base_scores, enhanced_scores = [], []
    for source, base, enhanced in zip(source_luma, base_images, enhanced_images):
        base_y = np.asarray(base.resize(
            (source.shape[1], source.shape[0]), Image.Resampling.BICUBIC)
            .convert('L'), dtype=np.uint8)
        enhanced_y = np.asarray(enhanced.convert('L'), dtype=np.uint8)
        base_scores.append(_psnr(source, base_y))
        enhanced_scores.append(_psnr(source, enhanced_y))
    base_psnr = float(np.mean(base_scores))
    enhanced_psnr = float(np.mean(enhanced_scores))

    for index in range(min(frame_count, 12)):
        base_images[index].save(out/f'base_{index:04d}.png')
        enhanced_images[index].save(out/f'enhanced_{index:04d}.png')
    wavfile.write(out/'source_audio.wav', v7.RATE, audio)
    wavfile.write(out/'watermarked_audio.wav', v7.RATE, encoded['audio'])
    stereo_output = np.column_stack((encoded['audio'], video_leg)).astype(
        np.float32)
    wavfile.write(out/'audio_left_video_right.wav', v7.RATE, stereo_output)

    duration = sample_count/v7.RATE
    data_bits_per_frame = detail_grid[0]*detail_grid[1]
    frame_delays = schedule['frame_ready_seconds'] - (
        np.arange(frame_count)*v7.PULSE_FRAME/v7.RATE)
    audio_score = _audio_metrics(
        audio, encoded['audio'], used_starts, BLOCK_SAMPLES)
    sweep_score = _sweep_reference_metrics()
    report = {
        'fixture': str(fixture),
        'routing': 'watermarked soundtrack left; ordinary mono V7 picture right',
        'channel_test': 'clean decoded-AAC PCM; no analog/tape impairment',
        'sample_rate': v7.RATE,
        'v7_frames': frame_count,
        'v7_packet_rate_fps': v7.PULSE_FPS,
        'duration_seconds': duration,
        'source_video_fps': source_fps,
        'audio_host': {
            'activity_rms_gate': float(activity_rms),
            'active_audio_blocks': int(len(active_starts)),
            'modulated_audio_blocks': int(len(used_starts)),
            'block_samples': BLOCK_SAMPLES,
            'block_hop_samples': BLOCK_HOP_SAMPLES,
            'audio_score': audio_score,
        },
        'modulation': {
            'scheme': 'active-block phase-QIM; bins reselected by received magnitude',
            'frequency_band_hz': [SWEEP_LOW_HZ, SWEEP_HIGH_HZ],
            'bins_per_block': bins_per_block,
            'bits_per_bin': bits_per_component,
            'phase_coset_periods': coset_periods,
            'max_phase_nudge_degrees': 180/coset_periods,
            'phase_step_degrees': 360/(coset_periods*
                                        (1 << bits_per_component)),
            'raw_active_capacity_bps': (
                len(active_starts)*bits_per_block/duration),
            'payload_bits_including_crc': int(len(payload_bits)),
            'gross_payload_bps': len(payload_bits)/duration,
            'detail_bits_per_frame': data_bits_per_frame,
            'crc_bits_per_frame': CRC_BYTES*8,
            'net_detail_bps': frame_count*data_bits_per_frame/duration,
            'payload_bytes_per_frame': len(payloads[0]),
            'padding_bits_in_final_block': (
                encoded['carrier_capacity_bits']-len(payload_bits)),
            'bit_errors': bit_errors,
            'ber': bit_errors/len(payload_bits) if len(payload_bits) else 0,
            'crc_valid_frames': crc_valid,
            'crc_total_frames': frame_count,
        },
        'video': {
            'base_displayable_packets': video_info.get('frames'),
            'eof_markers_validated': video_info.get('eof_markers_validated'),
            'high_resolution_luma_psnr_before_db': base_psnr,
            'high_resolution_luma_psnr_after_db': enhanced_psnr,
            'enhancement_gain_db': enhanced_psnr-base_psnr,
            'enhancement_correction_luma_codes': float(correction),
        },
        'enhancement_delivery_delay_seconds': {
            'median': float(np.median(frame_delays)),
            'p95': float(np.quantile(frame_delays, .95)),
            'maximum': float(np.max(frame_delays)),
        },
        'pulse_sweep_reference': sweep_score,
    }
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument('--out', type=Path,
                        default=Path('tmp/v7-audio-video-sidechannel'))
    parser.add_argument('--frames', type=int, default=None,
                        help='limit paired V7 frames (default: fixture duration)')
    parser.add_argument('--activity-rms', type=float, default=ACTIVE_AUDIO_RMS)
    parser.add_argument('--bins-per-block', type=int,
                        default=DEFAULT_BINS_PER_BLOCK)
    parser.add_argument('--bits-per-bin', type=int,
                        default=DEFAULT_BITS_PER_BIN)
    parser.add_argument('--coset-periods', type=int,
                        default=PHASE_COSET_PERIODS)
    args = parser.parse_args(argv)
    report = run(args.out, args.fixture, args.frames, args.activity_rms,
                 args.bins_per_block, args.bits_per_bin, args.coset_periods)
    print(json.dumps({
        'out': str(args.out),
        'frames': report['v7_frames'],
        'net_detail_bps': report['modulation']['net_detail_bps'],
        'gross_payload_bps': report['modulation']['gross_payload_bps'],
        'bit_errors': report['modulation']['bit_errors'],
        'crc_valid_frames': report['modulation']['crc_valid_frames'],
        'audio_snr_db': report['audio_host']['audio_score']['waveform_snr_db'],
        'base_psnr_db': report['video'][
            'high_resolution_luma_psnr_before_db'],
        'enhanced_psnr_db': report['video'][
            'high_resolution_luma_psnr_after_db'],
        'enhancement_gain_db': report['video']['enhancement_gain_db'],
        'delivery_delay_p95_s': report[
            'enhancement_delivery_delay_seconds']['p95'],
    }, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
