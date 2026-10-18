"""Compiled final-sample packet assembly and deterministic level fitting."""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def assemble_unmixed(shaped, header, eof, metadata, metadata_start,
                     body_gain, metadata_gain):
    """Match the float32 packet's staged additions, including float64 metadata."""
    body_gain = np.float32(body_gain)
    metadata_gain = np.float32(metadata_gain)
    packet = np.empty_like(shaped)
    for i in range(len(shaped)):
        for channel in range(shaped.shape[1]):
            value = np.float32(shaped[i, channel]*body_gain)
            value = np.float32(value+header[i, channel])
            if metadata_start <= i < metadata_start+len(metadata):
                value = np.float32(value+metadata[i-metadata_start]*metadata_gain)
            packet[i, channel] = np.float32(value+eof[i, channel])
    return packet


@njit(cache=True, nogil=True)
def _assemble(shaped, header, eof, metadata, overlay, metadata_start,
              body_gain, metadata_gain):
    packet = assemble_unmixed(shaped, header, eof, metadata, metadata_start,
                              body_gain, metadata_gain)
    gain = np.float32(body_gain)
    for i in range(len(packet)):
        for channel in range(packet.shape[1]):
            packet[i, channel] = np.float32(
                packet[i, channel]+np.float32(overlay[i, channel]*gain))
    return packet


@njit(cache=True, nogil=True)
def _peak(packet, start, stop):
    peak = 0.0
    for i in range(start, stop):
        for channel in range(packet.shape[1]):
            value = abs(packet[i, channel])
            if np.isnan(value):
                return np.nan
            peak = max(peak, value)
    return peak


@njit(cache=True, nogil=True)
def _trial_peak(shaped, header, eof, metadata, overlay, metadata_start,
                body_gain, metadata_gain, start, stop):
    """Measure a trial region directly, without allocating a trial packet."""
    gain = np.float32(body_gain)
    meta_gain = np.float32(metadata_gain)
    peak = 0.0
    for i in range(start, stop):
        for channel in range(shaped.shape[1]):
            value = np.float32(shaped[i, channel]*gain)
            value = np.float32(value+header[i, channel])
            if metadata_start <= i < metadata_start+len(metadata):
                value = np.float32(value+metadata[i-metadata_start]*meta_gain)
            value = np.float32(value+eof[i, channel])
            value = np.float32(value+np.float32(overlay[i, channel]*gain))
            magnitude = abs(value)
            if np.isnan(magnitude):
                return np.nan
            peak = max(peak, magnitude)
    return peak


@njit(cache=True, nogil=True)
def _trial_below(shaped, header, eof, metadata, overlay, metadata_start,
                 body_gain, body_start, body_stop, eof_start, body_ratio):
    header_peak = _trial_peak(shaped, header, eof, metadata, overlay, metadata_start,
                              body_gain, 1.0, 0, body_start)
    eof_peak = _trial_peak(shaped, header, eof, metadata, overlay, metadata_start,
                           body_gain, 1.0, eof_start, len(shaped))
    body_peak = _trial_peak(shaped, header, eof, metadata, overlay, metadata_start,
                            body_gain, 1.0, body_start, body_stop)
    return body_peak <= min(header_peak, eof_peak)*body_ratio


@njit(cache=True, nogil=True)
def limit_packet(shaped, header, eof, metadata, overlay, metadata_start,
                 body_start, body_stop, eof_start, body_ratio, metadata_ratio):
    """Same two 16-round gain searches as the wire's NumPy implementation."""
    body_gain = 1.0
    if not _trial_below(shaped, header, eof, metadata, overlay, metadata_start,
                         body_gain, body_start, body_stop, eof_start, body_ratio):
        low, high = 0.0, body_gain
        for _ in range(16):
            trial_gain = (low+high)*.5
            if _trial_below(shaped, header, eof, metadata, overlay, metadata_start,
                             trial_gain, body_start, body_stop, eof_start, body_ratio):
                low = trial_gain
            else:
                high = trial_gain
        body_gain = low
    packet = _assemble(shaped, header, eof, metadata, overlay, metadata_start,
                       body_gain, 1.0)
    metadata_limit = _peak(packet, body_start, body_stop)*metadata_ratio
    if _peak(packet, metadata_start, metadata_start+len(metadata)) > metadata_limit:
        low, high = 0.0, 1.0
        for _ in range(16):
            trial_gain = (low+high)*.5
            if _trial_peak(shaped, header, eof, metadata, overlay, metadata_start,
                            body_gain, trial_gain, metadata_start,
                            metadata_start+len(metadata)) <= metadata_limit:
                low = trial_gain
            else:
                high = trial_gain
        packet = _assemble(shaped, header, eof, metadata, overlay, metadata_start,
                           body_gain, low)
    return packet
