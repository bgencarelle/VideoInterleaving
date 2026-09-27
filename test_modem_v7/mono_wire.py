"""Standalone experimental V7 mono fold-off profile.

The mono body places a fresh protected head and body on 1,264 M-only slots,
then rotates the remaining coefficients through seven packet phases.  This
module stays in the V7 bench prototype until the profile and receiver contract
are validated.
"""
from contextlib import contextmanager
from dataclasses import replace

import numpy as np

from common import v7
import tone_code


MONO_GROUPS = 158
FRESH = 992
TAIL_PER = 272
HEAD_GROUPS = v7.HEAD//8
FRESH_GROUPS = FRESH//8
TAIL_GROUPS = TAIL_PER//8
MONO_MODE = tone_code.MONO_OFF


def mono_rank_tables(model):
    """Build seven rank maps for 992 fresh plus 272 rotating M slots."""
    order = np.asarray(model.order, dtype=int)
    if order.shape != (2880,):
        raise ValueError('mono V7 requires the canonical 2,880 coefficient set')
    tables = []
    for phase in range(v7.TAIL_PHASES):
        ranks = np.full((len(v7.GROUPS), 8), -1, dtype=int)
        ranks[:HEAD_GROUPS] = order[:v7.HEAD][v7.windows(HEAD_GROUPS)]
        body = order[v7.HEAD:FRESH]
        ranks[HEAD_GROUPS:FRESH_GROUPS] = body[v7.windows(
            FRESH_GROUPS-HEAD_GROUPS)]
        tail = order[FRESH+phase*TAIL_PER:FRESH+(phase+1)*TAIL_PER]
        if len(tail) < TAIL_PER:
            tail = np.pad(tail, (0, TAIL_PER-len(tail)),
                          constant_values=-1)
        ranks[FRESH_GROUPS:MONO_GROUPS] = tail[v7.windows(TAIL_GROUPS)]
        ranks.setflags(write=False)
        tables.append(ranks)
    return tuple(tables)


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
    """Sender/receiver glue for the experimental mono fold-off wire."""

    def __init__(self, model):
        self._models = {}
        self._base_models = {}
        self._installed = None
        self.model_for(model)

    def model_for(self, model):
        key = (model.encoding_type, id(model))
        if key not in self._models:
            ranks = mono_rank_tables(model)
            priors = tuple(v7.block_priors(model.gain, model.lam, rank)
                           for rank in ranks)
            mono_model = replace(
                model, rank_tables=ranks, block_prior_tables=priors,
                scale=float(model.scale*np.sqrt(2.0)))
            mono_model._mono_wire_profile = True
            self._models[key] = mono_model
            self._base_models[key] = model
        return self._models[key]

    def encode_packet(self, model, values, counter, aspect_code=0,
                      source_index=None, eof_marker=True):
        """Encode one mono-layout packet with a coded MONO_OFF status."""
        base = model
        mono_model = self.model_for(base)
        coeffs = base.coder.forward(values)/base.coder.gains
        if source_index is None:
            source_index = int(counter)-1
        from tools.v7_live import _encode_pulse_frame_coeffs
        packet = _encode_pulse_frame_coeffs(
            mono_model, coeffs, counter, aspect_code=aspect_code,
            source_index=source_index, eof_marker=eof_marker,
            pilot_values=MONO_PILOT_VALUES)
        return tone_code.add_tone_code(
            packet, int(counter), tone_code.encode_status(MONO_MODE))

    def encode(self, model, values, start_counter=1, aspect_codes=None,
               source_indices=None, eof_marker=True):
        values = list(values)
        codes = (list(aspect_codes) if aspect_codes is not None else
                 [0]*len(values))
        indexes = (list(source_indices) if source_indices is not None else
                   [start_counter+i-1 for i in range(len(values))])
        if not len(codes) == len(indexes) == len(values):
            raise ValueError('aspect_codes and source_indices must match values')
        return np.concatenate([
            self.encode_packet(model, value, start_counter+offset,
                               aspect_code=codes[offset],
                               source_index=indexes[offset],
                               eof_marker=eof_marker)
            for offset, value in enumerate(values)])

    @staticmethod
    def _status_for_body(model, body, force_float32=False):
        audio = np.asarray(body)
        if audio.shape != (v7.FRAME, 2):
            return None
        windows = audio.reshape(v7.F, v7.SYM, 2)[:, v7.WIN:v7.WIN+v7.N]
        Z = np.fft.rfft(windows, axis=1)*v7._receive_rotation(model)
        return tone_code.decode_tone_spectrum(Z, model)

    def install(self):
        """Gate decoding on MONO_OFF before image ranks/equalization are used."""
        if self._installed is not None:
            return
        real_decode = v7.decode_frame

        def decode_frame(model, x, tmap, counter, prev_tail, *args, **kwargs):
            body = kwargs.get('direct_body')
            if body is None:
                diag = {'displayable': False, 'held': True,
                        'mono_profile_rejected': 'body_unavailable'}
                return v7.Result(counter, 'lost', np.asarray(prev_tail).copy(),
                                 diag)
            status = self._status_for_body(model, body,
                                           kwargs.get('force_float32', False))
            if (not status['valid'] or
                    status['status']['mode'] != MONO_MODE):
                mode = (None if status['status'] is None else
                        status['status']['mode'])
                diag = {'displayable': False, 'held': True,
                        'coded_status_mode': mode,
                        'mono_profile_rejected': 'unknown_or_non_mono_status'}
                return v7.Result(counter, 'lost', np.asarray(prev_tail).copy(),
                                 diag)
            with mono_channel_profile():
                result = real_decode(model, x, tmap, counter, prev_tail,
                                     *args, **kwargs)
            if result is not None:
                timing = result.diag.get('pilot_timing') or {}
                if timing.get('coded_status_mode') != MONO_MODE:
                    return v7.Result(
                        counter, 'lost', np.asarray(prev_tail).copy(),
                        {'displayable': False, 'held': True,
                         'coded_status_mode': mode,
                         'mono_profile_rejected': 'receiver_status_mismatch'})
                result.diag['wire_profile'] = 'mono-fold-off'
            return result

        v7.decode_frame = decode_frame
        self._installed = real_decode

    def uninstall(self):
        if self._installed is None:
            return
        v7.decode_frame = self._installed
        self._installed = None

    @contextmanager
    def receiving(self):
        self.install()
        try:
            with tone_code.coded_pilot_timing():
                yield
        finally:
            self.uninstall()

    def values(self, model, result):
        return v7.values_from(model, result.coeffs)
