"""Experimental packet status channel on V7's two low pilot-tone bins.

This module is isolated from ``animation_modem``. It overlays the existing
bin-1/bin-3 tones on a tone-free pulse packet. Bin 1 stays steady. Across the
24 image OFDM symbols, bin 3 alternates 12 known BPSK pilot chips and 12 BPSK
status chips. The known pilot code and the steady bin-1 reference let the
receiver validate chip alignment and decode status before restoring the bin-3
phase track.

The 12-chip status is one of three length-12 codewords. Their pairwise Hamming
distance is six, so nearest-correlation decoding corrects two chip errors or
any five erased chips without a table ID, CRC, or soft-bit heuristic.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import threading

import numpy as np
from scipy.signal import savgol_filter

from common import RATE as CAPTURE_RATE, v7

FOLD_OFF = 0
FOLD_500 = 1
FOLD_1000 = 2
FOLD_MODE_TO_SLOTS = {FOLD_OFF: 0, FOLD_500: 500, FOLD_1000: 1000}

DATA_SYMBOLS = np.arange(1, v7.F, 2, dtype=int)
PILOT_SYMBOLS = np.arange(0, v7.F, 2, dtype=int)
# Balanced 12-chip code with maximum nonzero cyclic autocorrelation 4/12.
PILOT_CODE = np.asarray((1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1),
                        dtype=np.int8)
STATUS_BITS = 12
ALIGNMENT_MIN_SCORE = 0.60
TONE_MIN_BODY_RATIO = 0.03
STATUS_MIN_DISTANCE = 6
CHIP_SHAPE_SAMPLES = 8


def _repeat_four_chip_pattern(pattern):
    return tuple(int(bit) for _ in range(3) for bit in pattern)


# Three balanced codewords derived from distinct weight-two four-chip masks.
# Repeating the masks three times gives pairwise distance six in 12 chips.
STATUS_WORD_BY_MODE = {
    FOLD_OFF: _repeat_four_chip_pattern((0, 0, 1, 1)),
    FOLD_500: _repeat_four_chip_pattern((0, 1, 0, 1)),
    FOLD_1000: _repeat_four_chip_pattern((0, 1, 1, 0)),
}


def encode_status(mode):
    """Return the registered distance-six codeword for fold mode."""
    mode = int(mode)
    if mode not in FOLD_MODE_TO_SLOTS:
        raise ValueError('fold mode must be 0 (off), 1 (500), or 2 (1000)')
    return STATUS_WORD_BY_MODE[mode]


def decode_status(chips):
    """Nearest-correlation decode, bounded to two errors/five erasures.

    Erased chips are ``None``. The distance bound rejects any observation that
    is not uniquely inside one codeword's guaranteed decoding radius.
    """
    chips = tuple(chips)
    if len(chips) != STATUS_BITS or any(
            chip is not None and chip not in (0, 1) for chip in chips):
        return None
    erasures = sum(chip is None for chip in chips)
    candidates = []
    for word, candidate in STATUS_CODEBOOK:
        mode = candidate['mode']
        mismatches = tuple(index for index, (observed, expected) in
                           enumerate(zip(chips, word))
                           if observed is not None and int(observed) != expected)
        correlation = sum(1 if int(observed) == expected else -1
                          for observed, expected in zip(chips, word)
                          if observed is not None)
        candidates.append((correlation, len(mismatches), mode, mismatches))
    best_correlation = max(candidate[0] for candidate in candidates)
    best = [candidate for candidate in candidates
            if candidate[0] == best_correlation]
    if len(best) != 1:
        return None
    _, errors, mode, mismatches = best[0]
    if 2*errors+erasures >= STATUS_MIN_DISTANCE:
        return None
    return {'mode': mode, 'fold_slots': FOLD_MODE_TO_SLOTS[mode],
            'errors_corrected': errors, 'erasures_filled': erasures,
            'corrected_chips': list(mismatches)}


STATUS_CODEBOOK = tuple(
    (word, {'mode': mode, 'fold_slots': FOLD_MODE_TO_SLOTS[mode]})
    for mode, word in STATUS_WORD_BY_MODE.items())


def _packet_tone_amplitude(packet):
    body = np.asarray(packet)[
        v7.PULSE.SYNC_LEN:v7.PULSE.SYNC_LEN+v7.FRAME]
    body_rms = float(np.sqrt(np.mean(np.asarray(body, float)**2)))
    return math.sqrt(2.0)*body_rms*10**(v7.PILOT_TONE_REL_DB/20)


def add_tone_code(packet, counter, status_bits, pilot_code=PILOT_CODE):
    """Overlay the coded pilot/status tones on one tone-free V7 packet.

    ``pilot_code`` is carried on bin 3 at even image-symbol indices. Status
    bits are BPSK-keyed on bin 3 at odd indices. Bin 1 remains steady. Outside
    the 24-symbol image body bin 3 is steady as well. The tone phase advances
    continuously across packet counters just like V7's current tone helper.
    """
    packet = np.asarray(packet, dtype=np.float64)
    if packet.shape != (v7.PULSE_FRAME, 2):
        raise ValueError(f'packet must have shape ({v7.PULSE_FRAME}, 2)')
    status_bits = tuple(status_bits)
    if len(status_bits) != STATUS_BITS or any(bit not in (0, 1) for bit in status_bits):
        raise ValueError('status must contain exactly 12 binary chips')
    status_bits = tuple(int(bit) for bit in status_bits)
    pilot_code = np.asarray(pilot_code)
    if pilot_code.shape != (len(PILOT_SYMBOLS),) or not np.all(
            np.isin(pilot_code, (-1, 1))):
        raise ValueError('pilot code must contain 12 signs')
    packet_counter = int(counter)
    if packet_counter < 1 or packet_counter != counter:
        raise ValueError('packet counter must be one-based')
    pilot_code = pilot_code.astype(np.int8, copy=False)

    n = np.arange(v7.PULSE_FRAME, dtype=float)
    absolute = ((packet_counter-1)*v7.PILOT_TONE_DURATION+n) % v7.N
    phase1 = 2*np.pi*v7.PILOT_TONE_BINS[0]*absolute/v7.N
    phase3 = (2*np.pi*v7.PILOT_TONE_BINS[1]*absolute/v7.N +
              v7.PILOT_TONE_PHASES[v7.PILOT_TONE_BINS[1]])
    chips3 = np.ones(v7.PULSE_FRAME, dtype=float)
    body_start = v7.PULSE.SYNC_LEN
    symbol_chips = np.empty(v7.F, dtype=np.int8)
    for symbol in range(v7.F):
        symbol_chips[symbol] = (pilot_code[symbol//2] if symbol % 2 == 0 else
                                (1 if status_bits[symbol//2] == 0 else -1))
        start = body_start+symbol*v7.SYM
        chips3[start:start+v7.SYM] = symbol_chips[symbol]
        # Keep the coded phase transition inside the cyclic prefix. A raised
        # cosine from the preceding chip to this one avoids a hard discontinuity
        # in the added tone while leaving the useful FFT window unmodulated.
        shape = min(CHIP_SHAPE_SAMPLES, v7.CP)
        if shape > 1:
            previous = 1 if symbol == 0 else symbol_chips[symbol-1]
            x = np.arange(shape, dtype=float)/(shape-1)
            ramp = .5-.5*np.cos(np.pi*x)
            chips3[start:start+shape] = previous+(symbol_chips[symbol]-previous)*ramp
    amplitude = _packet_tone_amplitude(packet)
    tone = amplitude*(np.cos(phase1)+chips3*np.cos(phase3))
    return (packet+tone[:, None]).astype(np.float32)


def encode_packet(model, values, counter, mode=FOLD_OFF,
                  source_index=None, aspect_code=0,
                  pilot_code=PILOT_CODE, eof_marker=False):
    """Encode a normal V7 pulse packet and overlay time-coded pilots."""
    bits = encode_status(mode)
    packet = v7.encode_pulse_frame(
        model, values, counter, aspect_code=aspect_code,
        source_index=source_index, pilot_tones=False, eof_marker=eof_marker)
    return add_tone_code(packet, counter, bits, pilot_code=pilot_code)


def _symbol_phasors(samples, frame_start=0, frame_scale=None,
                    sample_rate=v7.RATE, timing_offsets=None):
    """Complex estimates of bins 0, 1, 2, and 3 for each of the 24 symbols.

    Uses direct fixed-window projections over the 128 useful samples (not an
    added general-purpose FFT path). ``frame_scale`` is capture samples per
    48 kHz reference sample, normally 1 at 48 kHz or 2 at 96 kHz.
    """
    audio = np.asarray(samples, dtype=float)
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2 and audio.shape[1] >= 1:
        mono = np.mean(audio, axis=1)
    else:
        raise ValueError('samples must be mono or multichannel audio')
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError('sample_rate must be positive and finite')
    scale = rate/v7.RATE if frame_scale is None else float(frame_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('frame_scale must be positive and finite')
    offsets = (np.zeros(v7.F) if timing_offsets is None else
               np.asarray(timing_offsets, dtype=float))
    if offsets.shape != (v7.F,) or not np.all(np.isfinite(offsets)):
        raise ValueError(f'timing_offsets must contain {v7.F} finite values')
    bins = (0, 1, 2, 3)
    phasors = np.zeros((v7.F, len(bins)), dtype=np.complex128)
    valid = np.zeros(v7.F, dtype=bool)
    sample_indexes = np.arange(len(mono))
    for symbol in range(v7.F):
        ref_start = v7.PULSE.SYNC_LEN+symbol*v7.SYM+v7.CP
        sample_count = max(1, int(round(v7.N*scale)))
        positions = (frame_start+(ref_start+offsets[symbol])*scale+
                     np.arange(sample_count))
        if positions[0] < 0 or positions[-1] >= len(mono):
            continue
        chunk = np.interp(positions, sample_indexes, mono)
        local_ref = np.arange(len(chunk), dtype=float)/scale
        for column, bin_index in enumerate(bins):
            carrier = np.exp(-2j*np.pi*bin_index*local_ref/v7.N)
            phasors[symbol, column] = 2*np.mean(chunk*carrier)
        valid[symbol] = True
    return bins, phasors, valid


def decode_tone_code(samples, frame_start=0, frame_scale=None,
                     sample_rate=v7.RATE,
                     pilot_code=PILOT_CODE,
                     alignment_min_score=ALIGNMENT_MIN_SCORE):
    """Decode status and reconstruct the bin-3 phase track from the coded tone.

    The steady bin-1 phasors supply an absolute per-symbol reference. Even
    bin-3 chips are de-spread with the known code; odd chips are decoded against
    the interpolated pilot phase. ``pilot_score`` is the code-alignment check.
    A missing tone, weak alignment, or an ambiguous codeword fails closed.
    """
    rate = float(sample_rate)
    scale = rate/v7.RATE if frame_scale is None else float(frame_scale)
    bins, z, valid = _symbol_phasors(samples, frame_start, scale, rate)
    code = np.asarray(pilot_code)
    if code.shape != (len(PILOT_SYMBOLS),):
        raise ValueError('pilot code must contain 12 signs')
    if not np.all(np.isin(code, (-1, 1))):
        raise ValueError('pilot code must contain 12 signs')
    code = code.astype(np.int8, copy=False)
    tone1 = z[:, bins.index(1)]
    empty = np.maximum(np.abs(z[:, bins.index(0)]),
                       np.abs(z[:, bins.index(2)]))
    n = int(np.count_nonzero(valid))
    if not n:
        return {'valid': False, 'reason': 'no_complete_symbols', 'status': None}

    # Pilot track values are local-symbol phasors. Remove the exact nominal
    # carrier advance between symbols before interpolation; bin 3 advances
    # 135 degrees per symbol, so unwrapping raw pilot samples would be
    # ambiguous across the two-symbol pilot spacing.
    symbol_index = np.arange(v7.F, dtype=float)
    bin1_step = v7.PILOT_TONE_BINS[0]*v7.SYM/v7.N
    bin3_step = v7.PILOT_TONE_BINS[1]*v7.SYM/v7.N
    d1_raw = tone1*np.exp(-2j*np.pi*bin1_step*symbol_index)
    body_lo = int(round(frame_start+v7.PULSE.SYNC_LEN*scale))
    body_hi = int(round(frame_start+(v7.PULSE.SYNC_LEN+v7.FRAME)*scale))
    body_audio = np.asarray(samples, dtype=float)[body_lo:body_hi]
    body_rms = float(np.sqrt(np.mean(body_audio*body_audio))) if body_audio.size else 0.0
    amplitude_floor = max(1e-8, body_rms*TONE_MIN_BODY_RATIO)

    def score_projection(projected):
        projected3 = projected[:, bins.index(3)]
        step_removed = projected3*np.exp(-2j*np.pi*bin3_step*symbol_index)
        amp = np.abs(projected3)
        median = float(np.median(amp[valid])) if np.any(valid) else 0.0
        active = (valid[PILOT_SYMBOLS] &
                  (amp[PILOT_SYMBOLS] >= amplitude_floor) &
                  (amp[PILOT_SYMBOLS] >= median*.20))
        positions = PILOT_SYMBOLS[active]
        signs = code[active]
        weights = amp[positions]
        units = step_removed[positions]/np.maximum(weights, 1e-12)
        score = float(abs(np.sum(units*signs*weights)) /
                      max(float(np.sum(weights)), 1e-12))
        return score, positions, amp, median

    # The first pass uses nominal symbol windows. Bin 1's residual phase is a
    # timing-error measurement; smooth it before shifting the symbol windows.
    # Select a correction only when it improves the independently known bin-3
    # code correlation. This leaves clean/hum-dominated packets on the raw
    # windows while recovering the flutter cases where uncorrected chips smear.
    candidates = [(0, np.zeros(v7.F), z)]
    phase1 = np.unwrap(np.angle(d1_raw))
    for window in (7, 9):
        smooth_phase = savgol_filter(phase1, window, 2, mode='interp')
        offsets = (-(smooth_phase-smooth_phase[0]) * v7.N /
                   (2*np.pi*v7.PILOT_TONE_BINS[0]))
        corrected = _symbol_phasors(samples, frame_start, scale,
                                     rate, timing_offsets=offsets)[1]
        candidates.append((window, offsets, corrected))
    choices = [(score_projection(projected)[0], window, offsets, projected)
               for window, offsets, projected in candidates]
    pilot_score, timing_window, timing_offsets, z = max(choices,
                                                        key=lambda item: item[0])
    tone1, tone3 = z[:, bins.index(1)], z[:, bins.index(3)]
    amp1, amp3 = np.abs(tone1), np.abs(tone3)
    d1 = tone1*np.exp(-2j*np.pi*bin1_step*symbol_index)
    d3 = tone3*np.exp(-2j*np.pi*bin3_step*symbol_index)
    _, pilot_positions, amp3, median3 = score_projection(z)
    pilot_active = np.isin(PILOT_SYMBOLS, pilot_positions)
    pilot_code_active = code[pilot_active]
    empty_floor = np.maximum(empty, body_rms*1e-5)
    snr_db = 20*np.log10(np.maximum(amp3, 1e-12)/empty_floor)
    tone_present = ((amp3 >= amplitude_floor) &
                    (amp3 >= median3*.20) & valid)
    bin1_present = (amp1 >= amplitude_floor) & valid

    if len(pilot_positions) >= 2:
        pilot_phase = np.unwrap(np.angle(d3[pilot_positions]*pilot_code_active))
        phase_at = np.interp(symbol_index, pilot_positions, pilot_phase)
        if pilot_positions[0] > 0:
            slope = (pilot_phase[1]-pilot_phase[0]) / (
                pilot_positions[1]-pilot_positions[0])
            phase_at[:pilot_positions[0]] = (
                pilot_phase[0] + slope*(symbol_index[:pilot_positions[0]]-
                                        pilot_positions[0]))
        if pilot_positions[-1] < v7.F-1:
            slope = (pilot_phase[-1]-pilot_phase[-2]) / (
                pilot_positions[-1]-pilot_positions[-2])
            phase_at[pilot_positions[-1]+1:] = (
                pilot_phase[-1] + slope*(symbol_index[pilot_positions[-1]+1:] -
                                         pilot_positions[-1]))
    else:
        phase_at = np.zeros(v7.F, dtype=float)
    odd = DATA_SYMBOLS
    data_metric = np.real(d3[odd]*np.exp(-1j*phase_at[odd]))
    data_confidence = np.abs(data_metric)/np.maximum(np.abs(d3[odd]), 1e-12)
    raw_bits = tuple(int(metric < 0) for metric in data_metric)
    data_erasures = tuple(index for index, symbol in enumerate(odd)
                          if not tone_present[symbol])
    observed_bits = tuple(None if index in data_erasures else bit
                          for index, bit in enumerate(raw_bits))
    status_candidate = decode_status(observed_bits)
    alignment_ok = (len(pilot_positions) >= 6 and
                    pilot_score >= float(alignment_min_score))
    status = status_candidate if alignment_ok else None
    bits = (encode_status(status['mode']) if status is not None else raw_bits)
    corrected_bits = [index for index, (actual, expected) in enumerate(
        zip(raw_bits, bits)) if index not in data_erasures and actual != expected]
    corrected_erasures = list(data_erasures) if status is not None else []

    recovered_signs = np.ones(v7.F, dtype=np.int8)
    recovered_signs[PILOT_SYMBOLS] = code
    recovered_signs[odd] = np.asarray([1 if bit == 0 else -1 for bit in bits],
                                      dtype=np.int8)
    bin3_track = np.unwrap(np.angle(d3*recovered_signs))
    bin1_track = np.unwrap(np.angle(d1))
    erasures = np.flatnonzero(~tone_present).tolist()
    if len(pilot_positions) < 6 or not alignment_ok:
        reason = 'pilot_code_misaligned'
    elif status is None:
        reason = 'status_crc_or_chip_error'
    else:
        reason = None
    return {
        'valid': bool(reason is None), 'reason': reason, 'status': status,
        'bits': bits, 'pilot_score': pilot_score,
        'pilot_amplitude': np.abs(tone3).tolist(),
        'tone_snr_db': snr_db.tolist(), 'tone_present': tone_present.tolist(),
        'bin1_present': bin1_present.tolist(), 'erasure_symbols': erasures,
        'pilot_symbols_used': pilot_positions.tolist(),
        'timing_window': timing_window,
        'timing_offsets': np.asarray(timing_offsets).tolist(),
        'data_confidence': data_confidence.tolist(),
        'raw_bits': raw_bits, 'corrected_bits': corrected_bits,
        'corrected_erasures': corrected_erasures,
        'status_errors_corrected': (status['errors_corrected']
                                    if status is not None else None),
        'status_erasures_filled': (status['erasures_filled']
                                   if status is not None else None),
        'recovered_signs': recovered_signs,
        'bin1_phase_track': bin1_track, 'bin3_phase_track': bin3_track,
        'bin1_phasors': d1, 'bin3_phasors': d3,
    }


def decode_tone_body(body, sample_rate=v7.RATE):
    """Decode coded pilots from V7's body-only, reference-rate sample block."""
    return decode_tone_code(
        body, frame_start=-v7.PULSE.SYNC_LEN, frame_scale=1.0,
        sample_rate=sample_rate)


@contextmanager
def coded_pilot_timing():
    """Temporarily despread valid coded chips before V7 estimates tone timing.

    ``decode_frame`` receives the exact sampled packet body. Decode its status
    there, then correct bin 3's known pilot and status signs in the input to
    V7's existing tone timing estimator. Image equalization still sees the
    original channel coefficients. This is prototype-only monkeypatching.
    """
    real_decode_frame = v7.decode_frame
    real_pilot_timing = v7.pilot_tone_timing
    local = threading.local()

    def pilot_tone_timing(Z, model, counter, include_metadata=False,
                          estimator='single'):
        signs = getattr(local, 'signs', None)
        expected_counter = getattr(local, 'counter', None)
        if signs is None or expected_counter != int(counter):
            return real_pilot_timing(
                Z, model, counter, include_metadata=include_metadata,
                estimator=estimator)
        adjusted = np.array(Z, copy=True)
        adjusted[:, 3, :] *= signs[:, None]
        track, metrics = real_pilot_timing(
            adjusted, model, counter, include_metadata=include_metadata,
            estimator=estimator)
        metrics = dict(metrics)
        metrics['coded_chips_removed'] = True
        metrics['coded_status_mode'] = getattr(local, 'mode', None)
        return track, metrics

    def decode_frame(*args, **kwargs):
        body = kwargs.get('direct_body')
        timing_mode = kwargs.get('pilot_timing', 'baseline')
        packet_counter = kwargs.get('pilot_counter')
        if packet_counter is None and len(args) > 3:
            packet_counter = args[3]
        old = (getattr(local, 'signs', None),
               getattr(local, 'counter', None),
               getattr(local, 'mode', None))
        local.signs = local.counter = local.mode = None
        if body is not None and timing_mode != 'baseline' and packet_counter is not None:
            decoded = decode_tone_body(body)
            if decoded['valid']:
                local.signs = np.asarray(decoded['recovered_signs'], dtype=float)
                local.counter = int(packet_counter)
                local.mode = decoded['status']['mode']
        try:
            return real_decode_frame(*args, **kwargs)
        finally:
            local.signs, local.counter, local.mode = old

    v7.decode_frame = decode_frame
    v7.pilot_tone_timing = pilot_tone_timing
    try:
        yield
    finally:
        v7.decode_frame = real_decode_frame
        v7.pilot_tone_timing = real_pilot_timing


def make_packet_stream(model, values, statuses, start_counter=1):
    """Convenience helper for status-coded V7 packet streams."""
    values, statuses = list(values), list(statuses)
    if len(values) != len(statuses):
        raise ValueError('values and statuses must have equal lengths')
    packets = []
    for offset, (value, status) in enumerate(zip(values, statuses)):
        counter = start_counter+offset
        mode = status[0] if isinstance(status, (tuple, list)) else status
        packets.append(encode_packet(model, value, counter, mode,
                                     source_index=offset))
    return np.concatenate(packets)


def acquire_packet_starts(samples, limit=None):
    """Find prototype packet boundaries with the designated pulse detector."""
    audio = np.asarray(samples)
    mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
    found = []
    cursor = 0
    while cursor+v7.PULSE.SYNC_LEN < len(mono):
        hit = v7.PULSE.measure_pulses(mono[cursor:])
        if hit is None:
            break
        edge, scale, confidence = hit
        frame_start = cursor+edge-16*scale
        if found and frame_start <= found[-1][0]:
            break
        frame_start = max(0.0, float(frame_start))
        found.append((frame_start, float(scale), float(confidence)))
        if limit is not None and len(found) >= limit:
            break
        # Skip past this packet's preamble while leaving margin for the next
        # pulse word if wow/flutter changes the local packet duration.
        cursor = max(cursor+1, int(frame_start+v7.PULSE_FRAME*scale*.90))
    return found


def match_tone_results_to_frames(results, starts, tone_results):
    """Pair separate tone scans to decoded packets by their frame-start time."""
    matched = {}
    used = set()
    for result in results:
        frame_start = result.diag.get('frame_start')
        if frame_start is None or not starts:
            continue
        choices = [(abs(float(frame_start)-float(start[0])), index)
                   for index, start in enumerate(starts) if index not in used]
        if not choices:
            continue
        distance, index = min(choices)
        scale = float(starts[index][1])
        tolerance = max(24*scale, .015*v7.PULSE_FRAME*scale)
        if distance <= tolerance:
            matched[id(result)] = tone_results[index]
            used.add(index)
    return matched, len(used)


def _selftest(args):
    from scipy.signal import resample_poly
    from common import CASES, impair, jitter_warp

    model = v7.load_model(.1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10)), 'box')
    frame = v7.REFERENCE_FIXTURE
    from PIL import Image
    with Image.open(frame) as image:
        values = v7.image_values(v7.prepare_image(image.convert('RGB'), 'box'),
                                 v7.V7_GRIDS, 'box')
    plain_packets = [v7.encode_pulse_frame(
        model, values, index+1, source_index=index, eof_marker=True)
        for index in range(args.packets)]
    coded_packets = [add_tone_code(packet, index+1,
                                   encode_status(FOLD_500))
                     for index, packet in enumerate(plain_packets)]
    steady_packets = [v7.encode_pulse_frame(
        model, values, index+1, source_index=index, pilot_tones=True,
        eof_marker=True) for index in range(args.packets)]
    streams = {
        'coded': np.concatenate(coded_packets),
        'steady': np.concatenate(steady_packets),
        'no_tone': np.concatenate(plain_packets),
    }
    cases = {case.name: case for case in CASES}
    for name in args.cases:
        jitter = (.001 if name == 'jitter 0.1%' else
                  .003 if name == 'jitter 0.3%' else 0.0)
        case = cases['clean-96k'] if jitter else cases[name]
        outputs = {}
        for kind, source in streams.items():
            wire = resample_poly(source, 2, 1, axis=0).astype(np.float32)
            if jitter:
                wire = jitter_warp(wire, jitter)
            damaged = impair(wire, case)
            hits = acquire_packet_starts(damaged, limit=args.packets)
            outputs[kind] = [decode_tone_code(
                damaged, frame_start=start, frame_scale=scale,
                sample_rate=CAPTURE_RATE) for start, scale, _ in hits]
        coded = outputs['coded']
        steady, no_tone = outputs['steady'], outputs['no_tone']
        valid = sum(result['valid'] for result in coded)
        correct = sum(result['valid'] and result['status']['mode'] == FOLD_500
                      for result in coded)
        scores = [result['pilot_score'] for result in coded]
        print(json.dumps({
            'case': name, 'acquired': len(coded), 'valid': valid,
            'correct_status': correct,
            'timing_corrections': sum(r['timing_window'] != 0 for r in coded),
            'soft_corrected_packets': sum(bool(r['corrected_bits']) for r in coded),
            'erasure_recovered_packets': sum(bool(r['corrected_erasures'])
                                             for r in coded),
            'steady_false_accepts': sum(r['valid'] for r in steady),
            'no_tone_false_accepts': sum(r['valid'] for r in no_tone),
            'mean_pilot_score': round(float(np.mean(scores)), 3) if scores else 0,
            'min_pilot_score': round(float(np.min(scores)), 3) if scores else 0,
            'steady_max_pilot_score': round(max(
                (r['pilot_score'] for r in steady), default=0), 3),
            'no_tone_max_pilot_score': round(max(
                (r['pilot_score'] for r in no_tone), default=0), 3),
        }, sort_keys=True), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    test = sub.add_parser('selftest', help='send/decode coded status on synthetic V7 packets')
    test.add_argument('--packets', type=int, default=12)
    test.add_argument('--cases', nargs='+', default=[
        'clean-96k', 'lowpass-10k', 'hiss-45', 'hiss-40', 'hiss-35',
        'dropouts', 'wow-flutter', 'fast-flutter', 'mains-buzz',
        'jitter 0.1%', 'jitter 0.3%'])
    args = parser.parse_args(argv)
    if args.packets < 1:
        parser.error('--packets must be positive')
    _selftest(args)


if __name__ == '__main__':
    main()
