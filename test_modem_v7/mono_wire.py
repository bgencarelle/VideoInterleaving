"""M-only pilot tables and status read shared by the V7 mono profiles.

The mono profiles put data and pilots on M only. This module holds the pilot
values their senders use, the receiver's M-only pilot expectations, and the
coded-status read of a packet body. The earlier rotating-tail mono wire that
lived here is retired.
"""
from contextlib import contextmanager

import numpy as np

from common import v7
import tone_code


def _profile_pilot_tables():
    """Return decoder pilot tables for a single M stream and zero S prior."""
    patterns = {
        key: np.asarray((value[0], 0j), dtype=complex)
        for key, value in v7.PATS.items()
    }
    pilot_values = np.asarray([patterns[key] for key in v7.PILOT_OBS])
    by_symbol = tuple(
        np.asarray([patterns[(s, int(b))] for b in bins])
        for s, bins in enumerate(v7.PILOT_BY_SYMBOL))
    pad_values = np.zeros_like(v7.PILOT_PAD_VALUES)
    for s in range(v7.F):
        for j, b in enumerate(v7.PILOT_PAD_BINS[s]):
            if v7.PILOT_PAD_VALID[s, j]:
                pad_values[s, j] = patterns[(s, int(b))]

    bin_values = np.zeros_like(v7._PILOT_BIN_VALUES)
    bin_ls = np.zeros_like(v7._PILOT_BIN_LS)
    for index, b in enumerate(v7._PILOT_BIN_FREQUENCIES):
        valid = v7._PILOT_BIN_VALID[index]
        symbols = v7._PILOT_BIN_SYMBOLS[index, valid]
        values = np.asarray([patterns[(int(s), int(b))] for s in symbols])
        bin_values[index, valid] = values
        bin_ls[index][:, valid] = np.linalg.pinv(values)

    return {
        'PATS': patterns,
        'PILOT_PV': pilot_values,
        'PILOT_PV32': pilot_values.astype(np.complex64),
        'PILOT_VALUES_BY_SYMBOL': by_symbol,
        'PILOT_PAD_VALUES': pad_values,
        'PILOT_PAD_VALUES32': pad_values.astype(np.complex64),
        '_PILOT_BIN_VALUES': bin_values,
        '_PILOT_BIN_VALUES32': bin_values.astype(np.complex64),
        '_PILOT_BIN_LS': bin_ls,
        '_PILOT_BIN_LS32': bin_ls.astype(np.complex64),
    }


MONO_PILOT_VALUES = np.zeros_like(v7.SCATTERED_PILOTS)
MONO_PILOT_VALUES[..., 0] = v7.SCATTERED_PILOTS[..., 0]
MONO_PILOT_VALUES.setflags(write=False)
MONO_RECEIVER_PILOTS = _profile_pilot_tables()


@contextmanager
def mono_channel_profile():
    """Temporarily use M-only pilot expectations in V7's existing receiver."""
    previous = {name: getattr(v7, name) for name in MONO_RECEIVER_PILOTS}
    try:
        for name, value in MONO_RECEIVER_PILOTS.items():
            setattr(v7, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(v7, name, value)


class MonoWire:
    """Coded-status read used by the mono profiles' receivers."""

    @staticmethod
    def _status_for_body(model, body, force_float32=False):
        audio = np.asarray(body)
        if audio.shape != (v7.FRAME, 2):
            return None
        windows = audio.reshape(v7.F, v7.SYM, 2)[:, v7.WIN:v7.WIN+v7.N]
        Z = np.fft.rfft(windows, axis=1)*v7._receive_rotation(model)
        return tone_code.decode_tone_spectrum(Z, model)
