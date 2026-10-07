"""Experimental all-fresh mono V7 wire with a 500-class corner fold.

Every packet carries the same 1,264 highest-ranked corner slots on M.  Five
hundred of those slots fold a lower-ranked corner coefficient into the host;
sixteen hosts are reserved for the table signature.  No coefficient history
is needed to reconstruct a current video frame.
"""
from contextlib import contextmanager
from dataclasses import replace
import threading

import numpy as np

from common import v7
from folding import FoldCodec, Grids
from mono_wire import MONO_PILOT_VALUES, mono_channel_profile, MonoWire
import tone_code


FRESH_SLOTS = 1264
FOLD_SLOTS = 500
SIGNATURE_SLOTS = 16
FOLDED_GUESTS = FOLD_SLOTS-SIGNATURE_SLOTS
MONO_GROUPS = FRESH_SLOTS//8
HEAD_GROUPS = v7.HEAD//8
MONO_VIDEO_MODE = tone_code.MONO_500
WIRE_PROFILE = 'mono-fresh-500'
MONO_FOLD_D = 1.1409647181002305
MONO_FOLD_TABLE_SHA256 = (
    'd075fac930d893b7bdbe9c4218aa45145ef634db06d297dee7ee21131916827d')
CHROMA_RANK_WEIGHT = 4.0
MONO_COLOUR_MODE = tone_code.MONO_1000
COLOUR_WIRE_PROFILE = 'mono-colour-500'
# Companded guests for the mono colour profiles (see folding.py): step, guest
# limit (model standard deviations), mu, and the symbol noise past which the
# guests are dropped. aspect-mono-500 uses the same settings. None restores
# linear guests clipped at +-2.5 (the earlier wire used those with D = 0.6).
MONO_COLOUR_COMPAND = {'step': 1.0, 'limit': 12.0, 'mu': 4.0,
                       'guest_noise_max': .15}
MONO_COLOUR_FOLD_D = MONO_COLOUR_COMPAND['step']
MONO_COLOUR_FOLD_TABLE_SHA256 = (
    '883b56bc5f759c3ed0a05e45d9d8f78d5649b819294282d0e87c537fa6cd5262')
_HOST_RANKS = (FRESH_SLOTS-FOLD_SLOTS, FRESH_SLOTS)
_GUEST_RANKS = (FRESH_SLOTS, FRESH_SLOTS+FOLD_SLOTS)


def fresh_rank_tables(model, order=None):
    """Build one fixed rank map: the same 1,264 coefficients every packet."""
    order = np.asarray(model.order if order is None else order, dtype=int)
    if order.shape != (2880,):
        raise ValueError('mono V7 requires the canonical 2,880 coefficient set')
    ranks = np.full((len(v7.GROUPS), 8), -1, dtype=int)
    ranks[:HEAD_GROUPS] = order[:v7.HEAD][v7.windows(HEAD_GROUPS)]
    ranks[HEAD_GROUPS:MONO_GROUPS] = order[v7.HEAD:FRESH_SLOTS][
        v7.windows(MONO_GROUPS-HEAD_GROUPS)]
    ranks.setflags(write=False)
    tables = []
    for _phase in range(v7.TAIL_PHASES):
        tables.append(ranks)
    return tuple(tables)


def colour_order(model):
    """model.order with the head kept and the rest re-ranked, chroma lam x4."""
    order = np.asarray(model.order, dtype=int)
    if order.shape != (2880,):
        raise ValueError('mono V7 requires the canonical 2,880 coefficient set')
    lam = np.asarray(model.lam, dtype=float).copy()
    plane = np.asarray(model.plane)
    lam[plane > 0] *= CHROMA_RANK_WEIGHT
    rest = order[v7.HEAD:]
    rest = rest[np.argsort(-lam[rest], kind='stable')]
    return np.concatenate([order[:v7.HEAD], rest])


def colour_fold_sets(model):
    """(order, hosts, guests): luma-only hosts and guests, as model indices."""
    order = colour_order(model)
    plane = np.asarray(model.plane)
    fresh = order[:FRESH_SLOTS]
    hosts = fresh[plane[fresh] == 0][-FOLD_SLOTS:]
    later = order[FRESH_SLOTS:]
    guests = later[plane[later] == 0][:FOLD_SLOTS]
    return order, hosts, guests


class MonoFreshFoldCodec(FoldCodec):
    """500-class fold within the fixed, fresh mono corner allocation."""

    def __init__(self, model, train_frames=None, D=MONO_FOLD_D,
                 design_db=30.0,
                 conf_min=.9, signature=SIGNATURE_SLOTS):
        if int(model.encoding_type) != v7.ENCODING_FILTER_CODES['box']:
            raise ValueError('mono 500-class folding requires the box model')
        self.model = model
        self.M = FOLD_SLOTS
        self.filter = 'box'
        self.conf_min = float(conf_min)
        self.design_db = float(design_db)
        self.fitted_on = 'v7_reference_face.png; mono fresh corner ranks'
        self.signature = int(signature)
        if self.signature != SIGNATURE_SLOTS:
            raise ValueError('mono 500-class folding requires 16 signature slots')
        self.noise_max = .3
        self.grid = Grids(v7.V7_GRIDS)
        self.kept = self.grid.corner_positions(v7.V7_SHAPES)

        # Preserve the protected 208-coefficient head.  The 500 weakest hosts
        # still in the fixed fresh prefix embed the next 500 corner ranks.
        host_rank = np.arange(*_HOST_RANKS, dtype=int)
        guest_rank = np.arange(*_GUEST_RANKS, dtype=int)
        self.sent_model_indices = np.asarray(model.order[:FRESH_SLOTS], dtype=int)
        self.hosts = np.asarray(model.order[host_rank], dtype=int)
        self.guest_model_indices = np.asarray(model.order[guest_rank], dtype=int)
        self.guests = self.kept[self.guest_model_indices]
        self.sd_host = np.sqrt(model.lam[self.hosts])
        self.sd_guest = np.sqrt(model.lam[self.guest_model_indices])

        if D is None:
            if train_frames is None:
                from PIL import Image
                with Image.open(v7.REFERENCE_FIXTURE) as image:
                    train_frames = [image.convert('RGB')]
            self.train = list(train_frames)
            D = self.best_step(self.design_db)
        else:
            self.train = list(train_frames or ())
        self.set_step(D)
        self._set_identity()
        if self.identity != MONO_FOLD_TABLE_SHA256:
            raise ValueError('mono 500-class fold table identity is not pinned')

    def table(self):
        table = super().table()
        table.update({
            'format': 'v7-mono-fresh-fold-1',
            'layout': WIRE_PROFILE,
            'fresh_slots': FRESH_SLOTS,
            'host_rank_range': list(_HOST_RANKS),
            'guest_rank_range': list(_GUEST_RANKS),
            'status_mode': MONO_VIDEO_MODE,
        })
        return table


class MonoFreshFoldWire:
    """Sender/receiver glue for the video-only mono 500-class profile."""

    requires_box = True
    disable_tail_memory = True
    wire_profile = WIRE_PROFILE
    fold_slots = FOLD_SLOTS
    status_mode = MONO_VIDEO_MODE
    use_fold = True
    codec_class = MonoFreshFoldCodec

    def coefficient_order(self, model):
        return np.asarray(model.order, dtype=int)

    def __init__(self, model, train_frames=None, side='both'):
        if side not in ('left', 'right', 'both'):
            raise ValueError("mono-video side must be 'left', 'right', or 'both'")
        self.side = side
        self.carrier_index = {'left': 0, 'right': 1}.get(side)
        self._models = {}
        self._base_models = {}
        self._codecs = {}
        self._codec_by_model_id = {}
        self._local = threading.local()
        self._installed = None
        self.train_frames = train_frames
        self.model_for(model)

    def model_for(self, model):
        if int(model.encoding_type) != v7.ENCODING_FILTER_CODES['box']:
            raise ValueError('mono 500-class folding requires the box model')
        key = (model.encoding_type, id(model))
        if key not in self._models:
            ranks = fresh_rank_tables(model, self.coefficient_order(model))
            priors = tuple(v7.block_priors(model.gain, model.lam, rank)
                           for rank in ranks)
            mono_model = replace(
                model, rank_tables=ranks, block_prior_tables=priors,
                scale=float(model.scale*np.sqrt(2.0)))
            mono_model._mono_wire_profile = self.wire_profile
            codec = (self.codec_class(model, train_frames=self.train_frames)
                     if self.use_fold else None)
            self._models[key] = mono_model
            self._base_models[key] = model
            self._codecs[key] = codec
            self._codec_by_model_id[id(mono_model)] = codec
        return self._models[key]

    def _codec(self, model):
        codec = self._codec_by_model_id.get(id(model))
        if codec is None:
            # Callers may hold the original model rather than its mono wrapper.
            key = (model.encoding_type, id(model))
            codec = self._codecs.get(key)
        if codec is None:
            raise ValueError('model is not registered for the mono video wire')
        return codec

    def _packet_model(self, model, aspect_code):
        """Mono model for one packet; layouts that follow the source aspect
        choose it from the packet's aspect code."""
        return self.model_for(model)

    def encode_packet(self, model, values, counter, aspect_code=0,
                      source_index=None):
        base = model
        mono_model = self._packet_model(base, aspect_code)
        codec = self._codec(mono_model) if self.use_fold else None
        coeffs = (codec.encode_coefficients(values) if self.use_fold else
                  base.coder.forward(values)/base.coder.gains)
        if source_index is None:
            source_index = int(counter)-1
        from tools.v7_live import _encode_pulse_frame_coeffs
        packet = _encode_pulse_frame_coeffs(
            mono_model, coeffs, counter, aspect_code=aspect_code,
            source_index=source_index,
            pilot_values=MONO_PILOT_VALUES,
            pulse_profile_code=self.status_mode)
        packet = tone_code.add_tone_code(
            packet, int(counter), tone_code.encode_status(self.status_mode))
        # Tone/status pilots are overlaid after pulse encoding and must also be
        # confined to the selected leg so the other output remains available.
        if self.side == 'left':
            packet[:, 1] = 0.0
        elif self.side == 'right':
            packet[:, 0] = 0.0
        return packet

    def encode(self, model, values, start_counter=1, aspect_codes=None,
               source_indices=None):
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
                               source_index=indexes[offset])
            for offset, value in enumerate(values)])

    def install(self):
        """Reject other layouts before equalization or tail-state updates."""
        if self._installed is not None:
            return
        real = (v7._equalize_numba, v7._equalize_numpy, v7.decode_frame)
        local = self._local

        def equalize_numba(model, Z, H, noise, counter):
            result = real[0](model, Z, H, noise, counter)
            local.equalized = (result[0].copy(), result[1].copy())
            return result

        def equalize_numpy(model, Z, H, noise, counter, force_float32=False):
            result = real[1](model, Z, H, noise, counter,
                             force_float32=force_float32)
            local.equalized = (result[0].copy(), result[1].copy())
            return result

        def decode_frame(model, x, tmap, counter, prev_tail, *args, **kwargs):
            local.equalized = None
            body = kwargs.get('direct_body')
            status = MonoWire._status_for_body(
                model, body, kwargs.get('force_float32', False))
            if (status is None or not status['valid'] or
                    status['status']['mode'] != self.status_mode):
                mode = (None if status is None or status['status'] is None else
                        status['status']['mode'])
                return v7.Result(
                    counter, 'lost', np.asarray(prev_tail).copy(),
                    {'displayable': False, 'held': True,
                     'coded_status_mode': mode,
                     'mono_profile_rejected': 'unknown_or_non_mono_status'})
            with mono_channel_profile():
                result = real[2](model, x, tmap, counter, prev_tail,
                                 *args, **kwargs)
            if result is not None:
                timing = result.diag.get('pilot_timing') or {}
                if timing.get('coded_status_mode') != self.status_mode:
                    return v7.Result(
                        counter, 'lost', np.asarray(prev_tail).copy(),
                        {'displayable': False, 'held': True,
                         'coded_status_mode': self.status_mode,
                         'mono_profile_rejected': 'receiver_status_mismatch'})
                if local.equalized is not None:
                    result.diag['mono_fold_eq'] = local.equalized
                result.diag['wire_profile'] = self.wire_profile
            return result

        v7._equalize_numba, v7._equalize_numpy, v7.decode_frame = (
            equalize_numba, equalize_numpy, decode_frame)
        self._installed = real

    def uninstall(self):
        if self._installed is None:
            return
        (v7._equalize_numba, v7._equalize_numpy,
         v7.decode_frame) = self._installed
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
        eq = result.diag.get('mono_fold_eq')
        if eq is None:
            return v7.values_from(model, result.coeffs)
        if not self.use_fold:
            return v7.values_from(model, result.coeffs)
        codec = self._codec(model)
        coeffs, xhat, conf = result.coeffs, eq[0], eq[1]
        fusion = result.diag.get('temporal_fusion')
        # A nested-fold codec is told the packet's tail slice; it answers
        # whether the packet's slot values may be averaged over packets.
        begin = getattr(codec, 'begin_packet', None)
        if begin is not None and begin(result.diag.get('tail_slice'), xhat, conf):
            fusion = None
        if fusion is not None:
            # Held pictures: average the equaliser output over packets
            # before the fold is undone (animation_modem/v7_temporal.py).
            from animation_modem.v7_temporal import fused_packet
            coeffs, xhat, conf = fused_packet(
                fusion, model, coeffs, xhat, conf,
                key=(self.wire_profile, result.diag.get('direction')))
            result.diag['fusion_gain'] = fusion.last_gain
        full = codec.decode(
            coeffs, xhat, conf, fallback=True,
            metadata_confirmed=True)
        # A nested fold's account of its coefficients, for the display.
        result.diag['luma_room'] = getattr(codec, 'last_room', None)
        return codec.grid.inverse(full)


class MonoColourFoldCodec(FoldCodec):
    """Luma-only 500-class fold over the colour-weighted fresh ranks."""

    def __init__(self, model, train_frames=None, D=MONO_COLOUR_FOLD_D,
                 design_db=40.0, conf_min=.9, signature=SIGNATURE_SLOTS,
                 check_identity=True):
        if int(model.encoding_type) != v7.ENCODING_FILTER_CODES['box']:
            raise ValueError('mono colour folding requires the box model')
        if int(signature) != SIGNATURE_SLOTS:
            raise ValueError('mono colour folding requires 16 signature slots')
        self.model = model
        self.M = FOLD_SLOTS
        self.filter = 'box'
        self.conf_min = float(conf_min)
        self.design_db = float(design_db)
        self.fitted_on = 'v7_reference_face.png; mono colour-weighted ranks'
        self.signature = int(signature)
        self.noise_max = .3
        self.grid = Grids(v7.V7_GRIDS)
        self.kept = self.grid.corner_positions(v7.V7_SHAPES)
        order, hosts, guests = colour_fold_sets(model)
        self.sent_model_indices = np.asarray(order[:FRESH_SLOTS], dtype=int)
        self.hosts = np.asarray(hosts, dtype=int)
        self.guest_model_indices = np.asarray(guests, dtype=int)
        self.guests = self.kept[self.guest_model_indices]
        self.sd_host = np.sqrt(model.lam[self.hosts])
        self.sd_guest = np.sqrt(model.lam[self.guest_model_indices])
        self.train = list(train_frames or ())
        self.set_step(D)
        if MONO_COLOUR_COMPAND is not None:
            self.use_compand(MONO_COLOUR_COMPAND['limit'],
                             MONO_COLOUR_COMPAND['mu'], D,
                             MONO_COLOUR_COMPAND['guest_noise_max'])
        self._set_identity()
        if check_identity and self.identity != MONO_COLOUR_FOLD_TABLE_SHA256:
            raise ValueError('mono colour fold table identity is not pinned')

    def table(self):
        table = FoldCodec.table(self)
        table.update({
            'format': 'v7-mono-colour-fold-1',
            'layout': COLOUR_WIRE_PROFILE,
            'fresh_slots': FRESH_SLOTS,
            'chroma_rank_weight': CHROMA_RANK_WEIGHT,
            'status_mode': MONO_COLOUR_MODE,
        })
        return table


class MonoColourFoldWire(MonoFreshFoldWire):
    """mono-fold-500 with colour-weighted ranks and a luma-only fold with
    companded guests (MONO_COLOUR_COMPAND)."""

    wire_profile = COLOUR_WIRE_PROFILE
    status_mode = MONO_COLOUR_MODE
    codec_class = MonoColourFoldCodec

    def coefficient_order(self, model):
        return colour_order(model)
