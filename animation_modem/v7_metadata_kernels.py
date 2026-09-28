"""Cached numeric kernels for V7 metadata demodulation."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def decode_metadata_spectrum(spectrum, phase, early, scale,
                             pilot_bins, data_bins):
    """Correct, combine, interpolate and hard-decision one metadata spectrum.

    The returned flag is false when fewer than two metadata pilots are usable;
    otherwise ``packed`` contains the same five bytes passed to the CRC parser.
    ``phase`` and ``early`` select the precision through their array dtype; the
    production default path uses the float64 model phase with the float32 FFT
    spectrum supplied by live audio.
    """
    observed = np.empty(spectrum.shape[0], dtype=phase.dtype)
    for b in range(spectrum.shape[0]):
        left = spectrum[b, 0]/scale
        left = left*np.conj(phase[b])
        left = left*early[b]
        if spectrum.shape[1] == 1:
            # Live mono capture is duplicated into M/S after resampling. Match
            # that shared observation here, before the body is available.
            right = left
        else:
            right = spectrum[b, 1]/scale
            right = right*np.conj(phase[b])
            right = right*early[b]
        observed[b] = left+right

    usable = 0
    for i in range(pilot_bins.shape[0]):
        if abs(observed[pilot_bins[i]]) > 1e-6:
            usable += 1
    packed = np.zeros(5, dtype=np.uint8)
    if usable < 2:
        return False, packed

    response = np.empty(data_bins.shape[0], dtype=phase.dtype)
    for i in range(data_bins.shape[0]):
        b = data_bins[i]
        hi = 1
        while hi < pilot_bins.shape[0]-1 and pilot_bins[hi] < b:
            hi += 1
        lo = hi-1
        fraction = ((b-pilot_bins[lo]) /
                    (pilot_bins[hi]-pilot_bins[lo]))
        low = observed[pilot_bins[lo]]
        high = observed[pilot_bins[hi]]
        real = low.real+(high.real-low.real)*fraction
        imag = low.imag+(high.imag-low.imag)*fraction
        response[i] = complex(real, imag)

        if abs(response[i]) > 1e-9:
            value = observed[b]/response[i]
        else:
            # np.divide(..., where=False) leaves its zero-initialized output;
            # zero compares nonnegative for both hard-decision bits.
            value = 0j
        bit = 2*i
        if value.real >= 0:
            packed[bit//8] |= np.uint8(1 << (7-bit % 8))
        bit += 1
        if value.imag >= 0:
            packed[bit//8] |= np.uint8(1 << (7-bit % 8))

    return True, packed
