#!/usr/bin/env python3
"""Stereo slices (experimental ``stereo-slices`` profile).

Each output channel is its own mono wire: the mono profiles' packet (framing,
M-only pilots, metadata, EOF marker, 1,264 slots, no tail rotation). Nothing
in one channel is needed to read the other, and each alone is a complete
picture.

Per plane the picture's coefficients nearest DC in picture frequency (the
aspect layouts' ordering) are cut in two: a *base* (1,020 luma, 110 per colour
plane) and, beyond it, as many *detail* coefficients. Every base coefficient
is paired with one detail coefficient, strongest base with weakest detail,
and one slot carries both:

    left  slot = base + DETAIL * detail
    right slot = base - DETAIL * detail

* both channels: the sum gives the base, the difference the detail: the
                 whole picture (2,040 luma coefficients);
* a mono sum:    the detail cancels: the base, a complete lower-detail
                 picture, exactly;
* one channel:   the base with the detail riding on it. DETAIL below 1 keeps
                 that disturbance small, and the pairing keeps it away from
                 the coefficients that matter most (those are paired with
                 the weakest detail).

Eight marker slots tell the three apart without a setting: a fixed pattern
sent positive on the left, negative on the right, so a sum reads zero. The
receiver averages the marker per input channel, so swapped cables do not
matter; the markers also measure crosstalk between the channels, which is
then removed.

Both channels carry the same colour (the base only), so the colour shown is
the same however many channels arrive.

An optional fold (FOLD_GUESTS, off as built) gives a channel heard alone
some of the detail: the weakest base slots then carry no detail and fold the
strongest detail coefficients as guests, the aspect mono profile's companded
fold, identical on both channels so a mono sum reads it too. It costs the
two-channel picture more than it gives one channel, so it is not used.

The receiver estimates the channels together: the base from both channels'
signal, the detail from their difference, each against the model's variance
for that coefficient.

The slots' statistics follow from the canonical model (base variance plus
DETAIL squared times the detail's) and are frozen in slice_tables.npz, whose
SHA-256 is pinned below.

Signalling: the aspect mono profile's coded status (MONO_OFF) with the
metadata model bit set to nearest, which that profile never sends. The layout
follows the packet's aspect code.

    python test_modem_v7/slice_wire.py build      # rebuild and print the pin
"""
import argparse
import hashlib
import os
import threading
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.fft import dct

from common import v7
import tone_code
from aspect_fold import (LAYOUT_CHOICES, LAYOUT_NAMES, AspectCoder,
                         _canonical_curves, _slug, _variance_fn,
                         layout_for_aspect_code, layout_size)
from mono_video import FRESH_SLOTS, HEAD_GROUPS, MONO_GROUPS
from mono_wire import MONO_PILOT_VALUES

PROFILE = 'stereo-slices'
STATUS_MODE = tone_code.MONO_OFF
ENCODING_TYPE = v7.ENCODING_FILTER_CODES['nearest']
TABLES = Path(__file__).resolve().with_name('slice_tables.npz')
TABLES_SHA256 = '991c133d81816ee77f5b6591210fd589c05b290ddf0566a78f1f704e726fe041'
TABLE_FORMAT = 'v7-stereo-slices-3'
MARKER_SLOTS = 8
SIGNATURE_SLOTS = 16
CHROMA_BASE = int(os.environ.get('SLICE_CHROMA', 110))                   # base (and detail) coefficients per colour plane
LUMA_BASE = FRESH_SLOTS-MARKER_SLOTS-SIGNATURE_SLOTS-2*CHROMA_BASE      # 988
BASE = (LUMA_BASE, CHROMA_BASE, CHROMA_BASE)
# The detail coefficient's strength in a slot. Lower: a channel heard alone
# is cleaner; with both channels the detail is restored but its noise grows
# by 1/DETAIL.
DETAIL = float(os.environ.get('SLICE_DETAIL', .35))
# Colour: with CHROMA_DETAIL the colour planes are split like luma; without,
# both channels carry the same colour (the base only), so the colour shown
# is the same however many channels arrive.
CHROMA_DETAIL = os.environ.get('SLICE_CHROMA_DETAIL', '0') == '1'
PLANE_DETAIL = (DETAIL, DETAIL if CHROMA_DETAIL else 0.0,
                DETAIL if CHROMA_DETAIL else 0.0)
MARKER_LEVEL = 3.0                  # marker size, in its slots' standard deviations
MARKER_PATTERN = np.array([1., -1., 1., 1., -1., 1., -1., -1.])
MARKER_MIN = .4                     # |score| at or above: one channel; below: a sum
# A receiver that names its input channel averages the marker over packets
# (this share of each new clean packet), so one bad packet cannot turn a
# channel into a "sum".
MARKER_BLEND = .25
# A packet with less head confidence than this, or with symbols lost to a
# splice or dropout, is damaged: it is not joined with a clean channel.
DAMAGED_HEAD_CONFIDENCE = .9
# Crosstalk between the channels is read from the markers and undone when
# both agree to CROSSTALK_AGREE and the share is within these bounds.
CROSSTALK_AGREE = .04
CROSSTALK_MIN, CROSSTALK_MAX = .02, .3
# The fold: guests per channel, the strongest slots skipped before the
# hosts, the step in host standard deviations, the guest's size as a share of
# the step, the compander's limit and mu, the signature's size in steps, the
# symbol noise (in steps) past which guests are dropped, and E compress(u)^2
# for real guests.
FOLD_GUESTS = int(os.environ.get('SLICE_GUESTS', 0))
FOLD_SKIP = 8
FOLD_STEP = float(os.environ.get('SLICE_STEP', 1.0))
FOLD_GUEST = .4
FOLD_LIMIT, FOLD_MU = 12.0, 4.0
SIGNATURE_STEPS = 3
SIGNATURE_PATTERN = np.array([1., 1., -1., 1., -1., -1., 1., -1.,
                              -1., 1., 1., -1., 1., -1., -1., 1.])
FOLD_GUEST_NOISE_MAX = .15
FOLD_SLIP_NOISE_MAX = .3              # past this the steps slip: hosts read plain
FOLD_COMPAND_POWER = .12
FOLD_POWER = 1+FOLD_STEP**2/12+(FOLD_GUEST*FOLD_STEP)**2*FOLD_COMPAND_POWER
UNKNOWN = 1e12                      # noise variance of a coefficient that was not read
TINY = 1e-30
CHROMA_RANK_WEIGHT = 4.0
SIDES = ('left', 'right')


def _compress(u):
    return (np.sign(u)*np.log1p(FOLD_MU*np.minimum(np.abs(u), FOLD_LIMIT) /
                                FOLD_LIMIT)/np.log1p(FOLD_MU))


def _expand(c):
    return np.sign(c)*FOLD_LIMIT*np.expm1(np.abs(c)*np.log1p(FOLD_MU))/FOLD_MU


@lru_cache(maxsize=16)
def _dct_matrix(size):
    matrix = np.ascontiguousarray(dct(np.eye(int(size)), axis=0, norm='ortho'))
    matrix.setflags(write=False)
    return matrix


# ------------------------------------------------------------------- build
def slice_positions(layout):
    """Per plane (base, detail) and the luma ring: positions within each
    plane's own grid, nearest DC first in the layout's picture frequency."""
    width, height = layout_size(layout)
    planes = []
    ring = None
    for plane, ((rows, cols), count) in enumerate(zip(v7.V7_GRIDS, BASE)):
        u, v = (axis.ravel().astype(np.int64)
                for axis in np.mgrid[:rows, :cols])
        key = u*u*width*width + v*v*height*height
        ordered = np.lexsort((v, u, key))
        need = 2*count
        # The base and detail must lie inside the grid; the ring may be cut
        # off by the grid's edge (it then takes the next nearest instead).
        if need > rows*cols or key[ordered[2*count-1]] >= min(
                key.reshape(rows, cols)[-1].min(),
                key.reshape(rows, cols)[:, -1].min()):
            raise ValueError(f'layout {layout} does not fit plane {plane}')
        planes.append((ordered[:count], ordered[count:2*count]))
    return planes, ring


def _layout_tables(layout, curves, phase):
    """One channel's slots for ``layout``.

    The coefficient vector is: luma slots, SIGNATURE_SLOTS filler, the
    marker, Cb slots, Cr slots. Slot k of a plane carries ``base[k]`` plus or
    minus DETAIL times ``detail[k]`` (grid positions, already paired).
    ``ring`` names the luma grid coefficients that ride as guests,
    ``ring_left`` / ``ring_right`` which of them each channel carries, and
    ``fold_hosts`` the vector indexes that carry them (strong luma slots,
    then the filler, which carries the signature).
    """
    variance = _variance_fn(curves, layout)
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    planes, ring = slice_positions(layout)
    base, detail, lam_base, lam_detail, ride = [], [], [], [], []
    for plane, (near, far) in enumerate(planes):
        near_lam = variance(offsets[plane]+near)
        far_lam = variance(offsets[plane]+far)
        # Strongest base with weakest detail: the coefficients that matter
        # most carry the least of the other.
        # Luma: the FOLD_GUESTS weakest base slots are fold hosts and carry
        # no detail; the strongest FOLD_GUESTS details are their guests (and
        # ride on a slot as well), the weakest FOLD_GUESTS details are not
        # sent.
        guests = FOLD_GUESTS if plane == 0 else 0
        by_base = np.argsort(-near_lam, kind='stable')
        by_detail = np.argsort(far_lam, kind='stable')
        by_detail = np.concatenate((by_detail[guests:], by_detail[:guests]))
        paired = np.empty(len(far), int)
        paired[by_base] = by_detail
        riding = np.zeros(len(far))
        riding[by_base[:len(far)-guests]] = PLANE_DETAIL[plane]
        ride.append(riding)
        if plane == 0:
            hosts = by_base[len(far)-guests:]
            ring = far[np.argsort(-far_lam, kind='stable')[:guests]]
        base.append(near)
        detail.append(far[paired])
        lam_base.append(near_lam)
        lam_detail.append(far_lam[paired])
    filler = SIGNATURE_SLOTS
    luma = slice(0, LUMA_BASE)
    start = LUMA_BASE+filler+MARKER_SLOTS
    chroma = (slice(start, start+CHROMA_BASE),
              slice(start+CHROMA_BASE, start+2*CHROMA_BASE))
    length = start+2*CHROMA_BASE
    lam = np.empty(length)
    lam_sum = np.zeros(length)
    riding = np.zeros(length)
    for part, near_lam, far_lam, strength in zip((luma, *chroma), lam_base,
                                                 lam_detail, ride):
        lam[part] = near_lam+strength**2*far_lam
        lam_sum[part] = near_lam
        riding[part] = strength
    # Fold hosts: the strongest luma slots after the first FOLD_SKIP.
    strong = np.asarray(hosts, int)
    # The filler carries the signature, which measures the fold's noise: it
    # is sent like the hosts it stands for. The marker is sent like a
    # stronger-than-average luma slot.
    lam[LUMA_BASE:LUMA_BASE+filler] = (np.median(lam[luma][strong])
                                      if len(strong) else np.median(lam[luma]))
    marker = LUMA_BASE+filler+np.arange(MARKER_SLOTS)
    lam[marker] = np.quantile(lam[luma], .75)
    plane = np.concatenate((np.zeros(start, int), np.full(CHROMA_BASE, 1),
                            np.full(CHROMA_BASE, 2)))
    # The coder's positions must be distinct grid cells; the spare slots sit
    # on the luma grid's far corner, beyond every ring.
    grid_rows, grid_cols = v7.V7_GRIDS[0]
    spare = grid_rows*grid_cols-1-np.arange(filler+MARKER_SLOTS)
    if np.intersect1d(spare, np.concatenate((base[0], detail[0]))).size:
        raise ValueError(f'no room for the spare slots of {layout}')
    positions = np.concatenate((base[0], spare, offsets[1]+base[1],
                                offsets[2]+base[2])).astype(np.int64)
    mu = np.zeros(length)
    for part, near, index in zip((luma, *chroma), base, range(3)):
        section = np.zeros(len(near))
        section[near == 0] = curves[index][3]
        mu[part] = section
    rank_lam = lam.copy()
    rank_lam[plane > 0] *= CHROMA_RANK_WEIGHT
    order = np.argsort(-rank_lam, kind='stable')
    gain = lam**-.25
    gain /= np.sqrt(np.mean(gain*gain*lam))
    tables = {'positions': positions, 'mu': mu, 'lam': lam, 'order': order,
              'gain': gain, 'marker': marker, 'lam_sum': lam_sum,
              'plane': plane,
              'base_0': base[0], 'base_1': base[1], 'base_2': base[2],
              'detail_0': detail[0], 'detail_1': detail[1],
              'detail_2': detail[2],
              'fold_hosts': np.concatenate(
                  (strong, LUMA_BASE+np.arange(filler))).astype(np.int64),
              'ring': ring.astype(np.int64),
              'ring_lam': (variance(offsets[0]+ring) if len(ring)
                           else np.zeros(0)),
              'ride': riding,
              'unit_rms': np.float64(1.0)}
    probe = _assemble(tables, phase, 1.0)
    synth = (np.random.default_rng(v7.LEVEL_SEED).standard_normal(length) *
             np.sqrt(lam) + mu)
    wave = v7.encode_frame_coeffs(probe, synth, 1,
                                  pilot_values=MONO_PILOT_VALUES)
    tables['unit_rms'] = np.float64(np.sqrt(np.mean(wave**2)))
    return tables


def build(_args=None):
    import io
    curves = _canonical_curves()
    phase = v7._frozen_tables()['phase']
    arrays = {}
    for layout in LAYOUT_NAMES:
        for key, value in _layout_tables(layout, curves, phase).items():
            arrays[f'{_slug(layout)}/{key}'] = np.asarray(value)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    blob = buffer.getvalue()
    TABLES.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    print(f"{TABLES.name}: {len(LAYOUT_NAMES)} layouts; pin TABLES_SHA256 = '{digest}'")
    return digest


# ------------------------------------------------------------------- model
@lru_cache(maxsize=1)
def _frozen():
    blob = TABLES.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != TABLES_SHA256:
        raise ValueError(f'{TABLES.name} SHA-256 {digest[:12]}… does not match the '
                         f'pinned {TABLES_SHA256[:12]}…')
    with np.load(TABLES, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


@lru_cache(maxsize=16)
def layout_tables(layout):
    if any(name in os.environ for name in
           ('SLICE_DETAIL', 'SLICE_GUESTS', 'SLICE_STEP', 'SLICE_CHROMA',
            'SLICE_CHROMA_DETAIL')):
        # A measurement run with other settings: built here, not frozen.
        return _layout_tables(layout, _canonical_curves(),
                              v7._frozen_tables()['phase'])
    prefix = f'{_slug(layout)}/'
    tables = {key[len(prefix):]: value for key, value in _frozen().items()
              if key.startswith(prefix)}
    if not tables:
        raise ValueError(f'no frozen slice tables for layout {layout!r}')
    return tables


def _rank_tables(order):
    """The mono wires' fixed rank map: the same slots in every packet."""
    order = np.asarray(order, dtype=int)
    ranks = np.full((len(v7.GROUPS), 8), -1, dtype=int)
    ranks[:HEAD_GROUPS] = order[:v7.HEAD][v7.windows(HEAD_GROUPS)]
    body = np.pad(order[v7.HEAD:FRESH_SLOTS],
                  (0, max(0, FRESH_SLOTS-len(order))), constant_values=-1)
    ranks[HEAD_GROUPS:MONO_GROUPS] = body[v7.windows(MONO_GROUPS-HEAD_GROUPS)]
    ranks.setflags(write=False)
    return tuple(ranks for _ in range(v7.TAIL_PHASES))


def _assemble(tables, phase, target_rms, template=None):
    order = np.asarray(tables['order'])
    lam = np.asarray(tables['lam'], float)
    gain = np.asarray(tables['gain'], float)
    mu = np.asarray(tables['mu'], float)
    head = np.zeros(len(lam), bool)
    head[order[:v7.HEAD]] = True
    ranks = _rank_tables(order)
    priors = tuple(v7.block_priors(gain, lam, idx) for idx in ranks)
    mu32, lam32, gain32 = (np.asarray(a, dtype=np.float32)
                           for a in (mu, lam, gain))
    phase32 = np.asarray(phase, dtype=np.complex64)
    priors32 = tuple(np.asarray(table, dtype=np.float32) for table in priors)
    for array in (mu32, lam32, gain32, phase32, *priors32):
        array.setflags(write=False)
    coder = AspectCoder(tables['positions'])
    scale = target_rms/float(tables['unit_rms'])
    plane = np.asarray(tables['plane'], int)
    if template is None:
        return v7.Model(coder, mu, lam, order, gain, phase, scale, plane,
                        head, ranks, ENCODING_TYPE, priors,
                        mu32, lam32, gain32, phase32, priors32)
    return replace(template, coder=coder, mu=mu, lam=lam, order=order,
                   gain=gain, scale=scale, plane=plane, head=head,
                   rank_tables=ranks, encoding_type=ENCODING_TYPE,
                   block_prior_tables=priors, mu32=mu32, lam32=lam32,
                   gain32=gain32, block_prior_tables32=priors32)


class _Slots:
    """One layout's tables as the arrays the per-frame code uses."""

    def __init__(self, layout, tables):
        self.layout = layout
        lam = np.asarray(tables['lam'], float)
        lam_sum = np.asarray(tables['lam_sum'], float)
        self.length = len(lam)
        self.model_lam = lam
        self.model_mu = np.asarray(tables['mu'], float)
        self.marker = np.asarray(tables['marker'], int)
        self.marker_sd = np.sqrt(lam[self.marker])
        self.marker_values = MARKER_LEVEL*MARKER_PATTERN*self.marker_sd
        start = int(self.marker[-1])+1
        self.parts = (slice(0, LUMA_BASE),
                      slice(start, start+CHROMA_BASE),
                      slice(start+CHROMA_BASE, start+2*CHROMA_BASE))
        self.base = tuple(np.asarray(tables[f'base_{plane}'], int)
                          for plane in range(3))
        self.detail = tuple(np.asarray(tables[f'detail_{plane}'], int)
                            for plane in range(3))
        self.mu = tuple(self.model_mu[part] for part in self.parts)
        # Per plane: the base's variance and, in slot units, the detail's.
        self.common = tuple(lam_sum[part] for part in self.parts)
        self.differ = tuple(np.maximum(lam[part]-lam_sum[part], 0.0)
                            for part in self.parts)
        self.lam = tuple(lam[part] for part in self.parts)
        self.fold_hosts = np.asarray(tables['fold_hosts'], int)
        self.guests = len(self.fold_hosts)-SIGNATURE_SLOTS
        self.sd_host = np.sqrt(lam[self.fold_hosts])
        self.ring = np.asarray(tables['ring'], int)
        self.ring_lam = np.asarray(tables['ring_lam'], float)
        self.sd_ring = np.sqrt(self.ring_lam)
        self.ride = tuple(np.asarray(tables['ride'], float)[part]
                          for part in self.parts)
        self.sent = tuple(
            np.concatenate((self.base[plane], self.detail[plane]))
            if self.ride[plane].any() else self.base[plane] for plane in range(3))


class Half:
    """One decoded channel: what the marker says it is, and its slots before
    any shrinking (``raw``: per plane, values less their means and their
    noise variances) with the ring coefficients it carried (``ring``: values
    and noise variances over the whole ring)."""

    __slots__ = ('kind', 'score', 'layout', 'good', 'noise', 'damaged',
                 'steady', 'raw', 'ring')

    def __init__(self, kind, score, layout, good=True, noise=0.0,
                 damaged=False, steady=None, raw=None, ring=None):
        self.kind, self.score, self.layout = kind, score, layout
        self.good, self.noise = bool(good), float(noise)
        self.damaged = bool(damaged)
        self.steady = score if steady is None else float(steady)
        self.raw, self.ring = raw, ring


class SliceWire:
    """Sender/receiver glue for the stereo slices profile."""

    status_mode = STATUS_MODE
    pulse_profile_code = STATUS_MODE
    wire_profile = PROFILE
    requires_box = True
    disable_tail_memory = True
    fold_slots = 0

    def __init__(self, layout='auto'):
        if layout not in LAYOUT_CHOICES:
            raise ValueError(f'unknown aspect layout {layout!r}; choose {LAYOUT_CHOICES}')
        self.layout = layout
        self._models = {}
        self._slots = {}
        self._layout_of = {}
        self._marker = {}
        self._masks = {}
        self._lock = threading.Lock()

    def prepare(self, _base_model=None):
        return self

    def reset(self):
        """Forget what each input channel was carrying."""
        self._marker.clear()

    def layout_for(self, aspect_code):
        if self.layout != 'auto':
            return self.layout
        return layout_for_aspect_code(aspect_code)

    def slots(self, layout):
        if layout not in self._slots:
            self._slots[layout] = _Slots(layout, layout_tables(layout))
        return self._slots[layout]

    def model_for(self, base_model, layout):
        """One channel's model for ``layout`` (either canonical base model)."""
        if id(base_model) in self._layout_of:
            return base_model                           # already a slice model
        key = (id(base_model), layout)
        with self._lock:
            if key not in self._models:
                name = v7.ENCODING_FILTERS[int(base_model.encoding_type)]
                canonical = float(v7._frozen_tables()[f'{name}/unit_rms'])
                model = _assemble(layout_tables(layout), base_model.phase,
                                  float(base_model.scale)*canonical,
                                  template=base_model)
                model._mono_wire_profile = self.wire_profile
                self._models[key] = model
                self._layout_of[id(model)] = layout
                self.slots(layout)
            return self._models[key]

    def luma_sent_mask(self, layout):
        """The luma grid coefficients both channels carry between them."""
        return self._sent_masks(layout)[0]

    def chroma_sent_masks(self, layout):
        """Per chroma grid: the coefficients both channels carry."""
        return list(self._sent_masks(layout)[1:])

    def _sent_masks(self, layout):
        if layout not in self._masks:
            slots = self.slots(layout)
            masks = []
            for sent, (rows, cols) in zip(slots.sent, v7.V7_GRIDS):
                mask = np.zeros(rows*cols, bool)
                mask[sent] = True
                mask.setflags(write=False)
                masks.append(mask)
            self._masks[layout] = tuple(masks)
        return self._masks[layout]

    # ---------------------------------------------------------------- sender
    @staticmethod
    def spectra(values):
        """Grid values -> each plane's grid coefficients (flattened)."""
        values = np.asarray(values, float)
        out, offset = [], 0
        for rows, cols in v7.V7_GRIDS:
            grid = np.ascontiguousarray(
                values[offset:offset+rows*cols].reshape(rows, cols))
            out.append((_dct_matrix(rows) @ grid @ _dct_matrix(cols).T).ravel())
            offset += rows*cols
        return out

    def channel_coefficients(self, model, spectra, kind):
        """One channel's coefficient vector from the picture's ``spectra``;
        ``kind`` 'left' or 'right'."""
        if kind not in SIDES:
            raise ValueError("a channel carries the 'left' or 'right' slots")
        slots = self.slots(self._layout_of[id(model)])
        sign = 1.0 if kind == 'left' else -1.0
        out = np.zeros(slots.length)
        for part, spectrum, base, detail, strength in zip(
                slots.parts, spectra, slots.base, slots.detail, slots.ride):
            out[part] = spectrum[base]+sign*strength*spectrum[detail]
        out[slots.marker] = sign*slots.marker_values
        hosts = slots.fold_hosts
        mu = slots.model_mu[hosts]
        symbol = np.zeros(len(hosts))
        if slots.guests:
            real = slice(0, slots.guests)
            symbol[real] = (
                FOLD_STEP*np.round((out[hosts[real]]-mu[real]) /
                                   slots.sd_host[real]/FOLD_STEP) +
                FOLD_GUEST*FOLD_STEP*_compress(
                    spectra[0][slots.ring]/slots.sd_ring))
        symbol[slots.guests:] = SIGNATURE_STEPS*FOLD_STEP*SIGNATURE_PATTERN
        out[hosts] = mu+slots.sd_host*symbol/np.sqrt(FOLD_POWER)
        return out

    def encode_packet(self, base_model, values, counter, aspect_code=0,
                      source_index=None, sides=SIDES):
        """One stereo packet: ``sides`` names what each output channel
        carries ('left', 'right', or None for silence)."""
        layout = self.layout_for(aspect_code)
        if layout is None:
            raise ValueError(f'no aspect layout for aspect code {aspect_code!r}')
        model = self.model_for(base_model, layout)
        if source_index is None:
            source_index = int(counter)-1
        carried = [kind for kind in sides if kind is not None]
        if not carried or len(sides) != 2:
            raise ValueError('sides names what each of two channels carries')
        spectra = self.spectra(values)
        coefficients = [self.channel_coefficients(model, spectra, kind)
                        for kind in carried]
        # One packet build: with two channels each is its own mono wire.
        packet = v7.encode_pulse_frame_coeffs(
            model, coefficients[0], counter, aspect_code=aspect_code,
            source_index=source_index, pilot_tones=False,
            pilot_values=MONO_PILOT_VALUES,
            pulse_profile_code=self.pulse_profile_code,
            right_coeffs=coefficients[1] if len(carried) == 2 else None)
        packet = tone_code.add_tone_code(
            packet, int(counter), tone_code.encode_status(self.status_mode))
        for index, kind in enumerate(sides):
            if kind is None:
                packet[:, index] = 0.0
        return packet

    def encode(self, base_model, values, start_counter=1, aspect_codes=None,
               source_indices=None, sides=SIDES):
        values = list(values)
        codes = (list(aspect_codes) if aspect_codes is not None else
                 [0]*len(values))
        indexes = (list(source_indices) if source_indices is not None else
                   [start_counter+i-1 for i in range(len(values))])
        if not len(codes) == len(indexes) == len(values):
            raise ValueError('aspect_codes and source_indices must match values')
        return np.concatenate([
            self.encode_packet(base_model, value, start_counter+offset,
                               aspect_code=codes[offset],
                               source_index=indexes[offset], sides=sides)
            for offset, value in enumerate(values)])

    # -------------------------------------------------------------- receiver
    def half(self, model, result, channel=None):
        """The decoded channel of ``result`` as a Half. ``channel`` names the
        receiver's input channel, whose marker is then averaged over packets."""
        layout = self._layout_of[id(model)]
        slots = self.slots(layout)
        equalized = result.diag.get('mono_fold_eq')
        if equalized is not None:
            # The equaliser's own output over its confidence: its shrinking
            # of noisy slots is undone and kept as each slot's noise.
            trust = np.clip(np.asarray(equalized[1], float), 1e-3, 1.0)
            seen = np.asarray(equalized[0], float)/trust
            noise = slots.model_lam*(1-trust)/trust
        else:
            seen = np.asarray(result.coeffs, float)-slots.model_mu
            noise = np.zeros(slots.length)
        score = float(np.mean(seen[slots.marker]/slots.marker_sd *
                              MARKER_PATTERN)/MARKER_LEVEL)
        diag = result.diag
        damaged = float(diag.get('head_confidence') or 0) < DAMAGED_HEAD_CONFIDENCE
        for key in ('splice', 'erased_symbols'):
            detail = diag.get(key)
            if isinstance(detail, dict) and detail.get('damaged_symbols'):
                damaged = True
        steady = score
        if channel is not None:
            previous = self._marker.get(channel)
            if previous is None:
                steady = score
            elif damaged or result.status == 'lost':
                steady = previous
            else:
                steady = previous+MARKER_BLEND*(score-previous)
            if not damaged or previous is None:
                self._marker[channel] = steady
        kind = ('left' if steady >= MARKER_MIN else
                'right' if steady <= -MARKER_MIN else 'sum')
        raw = [(seen[part].copy(), noise[part].copy()) for part in slots.parts]
        ring = None
        if slots.guests:
            ring = (np.zeros(len(slots.ring)), np.full(len(slots.ring), UNKNOWN))
            self._unfold(slots, seen, noise, kind, raw[0], ring)
        pilot = result.diag.get('noise') or [0.0]
        return Half(kind, score, layout, result.status != 'lost', max(pilot),
                    damaged, steady, raw, ring)

    @staticmethod
    def _unfold(slots, seen, noise, kind, luma, ring):
        """Read the folded slots (the same on both channels, so a sum reads
        them too): the hosts' steps into the luma slots, the guests into
        ``ring`` (values, noise variances). When the signature's noise says
        the steps would slip, the hosts are read plain and no guests."""
        hosts = slots.fold_hosts
        real = slice(0, slots.guests)
        sd = slots.sd_host
        symbol = seen[hosts]/sd*np.sqrt(FOLD_POWER)
        amp = FOLD_GUEST*FOLD_STEP
        carried = amp**2*FOLD_COMPAND_POWER
        cells = hosts[real]                              # luma slots are first
        resid = symbol[slots.guests:]-SIGNATURE_STEPS*FOLD_STEP*SIGNATURE_PATTERN
        sigma = float(np.sqrt(np.mean(resid**2)))
        if sigma > FOLD_SLIP_NOISE_MAX*FOLD_STEP:
            luma[0][cells] = sd[real]*symbol[real]
            luma[1][cells] = noise[cells]+sd[real]**2*(FOLD_STEP**2/12+carried)
            return
        steps = np.round(symbol[real]/FOLD_STEP)
        luma[0][cells] = sd[real]*FOLD_STEP*steps
        luma[1][cells] = sd[real]**2*sigma**2+TINY
        if sigma <= FOLD_GUEST_NOISE_MAX*FOLD_STEP:
            residual = np.clip((symbol[real]-FOLD_STEP*steps)/amp, -1.0, 1.0)
            ring[0][:] = _expand(residual)*slots.sd_ring
            ring[1][:] = slots.ring_lam*sigma**2/carried+TINY

    @staticmethod
    def choose(halves):
        """The halves to show: a clean channel is not joined with a damaged
        one (a noisy one is: both channels of a tape are equally noisy)."""
        halves = [half for half in halves if half is not None]
        if len(halves) < 2:
            return halves
        first, second = halves
        if first.layout != second.layout:
            return [first if first.good or not second.good else second]
        if first.damaged != second.damaged:
            return [second if first.damaged else first]
        if first.kind == second.kind and first.kind != 'sum':
            # The same slots twice (one channel wired to both inputs).
            return [first if first.noise <= second.noise else second]
        return halves

    @staticmethod
    def crosstalk(halves):
        """The share of each channel heard in the other, from the markers.

        With a share e of the other channel mixed in, a channel's pilots
        grow by 1+e while its marker (opposite on the other channel) shrinks
        to 1-e, so the marker reads r = (1-e)/(1+e). Used only when both
        channels' averaged markers agree; small or implausible values are
        taken as none."""
        sizes = [abs(half.steady) for half in halves]
        if len(sizes) != 2 or abs(sizes[0]-sizes[1]) > CROSSTALK_AGREE:
            return 0.0
        ratio = sum(sizes)/2
        leak = (1-ratio)/(1+ratio)
        return float(leak) if CROSSTALK_MIN <= abs(leak) <= CROSSTALK_MAX else 0.0

    def estimate(self, halves):
        """Per plane (base, detail) coefficient estimates from the chosen
        channels, the base with its mean added (detail None for a sum).

        A slot is the base plus or minus DETAIL times the detail, and the
        tables give the variance of each. With both channels the base is
        estimated from their mean and the detail from half their difference;
        a mono sum is the base alone; one channel is split between the two
        by their variances (mostly base, where it matters)."""
        slots = self.slots(halves[0].layout)
        kinds = [half.kind for half in halves]
        sums = [half for half in halves if half.kind == 'sum']
        out = []
        if sorted(kinds) == sorted(SIDES):
            leak = self.crosstalk(halves)
            by = {half.kind: half.raw for half in halves}
            for plane in range(3):
                (a, a_noise), (b, b_noise) = by['left'][plane], by['right'][plane]
                if leak:
                    a, b = (a-leak*b)/(1-leak), (b-leak*a)/(1-leak)
                noise = (a_noise+b_noise)/4+TINY
                common, differ = slots.common[plane], slots.differ[plane]
                strength = slots.ride[plane]
                out.append(((a+b)/2*common/(common+noise)+slots.mu[plane],
                            (a-b)/2*differ/(differ+noise) /
                            np.where(strength > 0, strength, 1.0)
                            if strength.any() else None))
            return out
        if sums:
            for plane in range(3):
                seen = sum(half.raw[plane][0] for half in sums)/len(sums)
                noise = sum(half.raw[plane][1] for half in sums)/len(sums)**2
                common = slots.common[plane]
                out.append((seen*common/(common+noise+TINY)+slots.mu[plane],
                            None))
            return out
        sign = 1.0 if halves[0].kind == 'left' else -1.0
        for plane in range(3):
            seen, noise = halves[0].raw[plane]
            total = slots.lam[plane]+noise+TINY
            strength = slots.ride[plane]
            out.append((seen*slots.common[plane]/total+slots.mu[plane],
                        sign*seen*slots.differ[plane]/total /
                        np.where(strength > 0, strength, 1.0)
                        if strength.any() else None))
        return out

    def values(self, halves):
        """Grid values for display from one or two decoded channels."""
        halves = self.choose(halves)
        slots = self.slots(halves[0].layout)
        # With both channels the guests' coefficients are read from their
        # slots, far more cleanly than from the fold.
        both = sorted(half.kind for half in halves) == sorted(SIDES)
        ring = None if both else self._ring(slots, halves)
        out = []
        for plane, ((base, detail), (rows, cols)) in enumerate(
                zip(self.estimate(halves), v7.V7_GRIDS)):
            spectrum = np.zeros(rows*cols)
            spectrum[slots.base[plane]] = base
            if detail is not None:
                spectrum[slots.detail[plane]] = detail
            if plane == 0 and ring is not None:
                spectrum[slots.ring] = ring
            out.append((_dct_matrix(rows).T @ spectrum.reshape(rows, cols) @
                        _dct_matrix(cols)).ravel())
        return np.concatenate(out)

    @staticmethod
    def _ring(slots, halves):
        """The ring's coefficients from every channel that carried them:
        each channel's reading weighted by its noise, then shrunk by what the
        model expects of that coefficient. None when nothing was read."""
        readings = [half.ring for half in halves if half.ring is not None]
        if not readings:
            return None
        weight = sum(1.0/noise for _, noise in readings)
        if not np.any(weight > 2.0/UNKNOWN):
            return None
        value = sum(values/noise for values, noise in readings)/weight
        return value*slots.ring_lam/(slots.ring_lam+1.0/weight)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build',))
    build(parser.parse_args(argv))


if __name__ == '__main__':
    main()

