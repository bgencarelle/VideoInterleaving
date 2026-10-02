#!/usr/bin/env python3
"""Stereo slices (experimental ``stereo-slices`` profile).

Each output channel is its own mono wire: the mono profiles' packet (framing,
M-only pilots, metadata, EOF marker, 1,264 slots, no tail rotation) with every
slot sent plainly (no fold). Nothing in one channel is needed to read the
other.

The picture is a luma rectangle P (rows x cols, cols even) and a smaller
chroma rectangle per colour plane, shaped to the layout. The left channel
carries one checkerboard half of each (sample (i, j) with i+j even), the
right channel the other, each as a rows x cols/2 array whose own DCT is what
the slots hold. So

* both channels: the halves interleave into the whole picture;
* one channel:   the missing samples are filled from their four neighbours,
                 a complete picture that has lost a little in every direction;
* a mono sum:    the two arrays add, which is the picture averaged over
                 horizontal pairs, again complete.

Eight marker slots tell the three apart without a setting: a fixed pattern
sent positive on the left, negative on the right, so a sum reads zero.

The slots' statistics (a half's DCT mixes the picture's low frequencies with
their checkerboard aliases) are synthesised from the canonical model and
frozen in slice_tables.npz, whose SHA-256 is pinned below.

Signalling: the aspect mono profile's coded status (MONO_OFF) with the
metadata model bit set to nearest, which that profile never sends. The layout
follows the packet's aspect code.

    python test_modem_v7/slice_wire.py build      # rebuild and print the pin
"""
import argparse
import hashlib
import threading
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn

from common import v7
import tone_code
from aspect_fold import (LAYOUT_CHOICES, LAYOUT_NAMES, AspectCoder,
                         _canonical_curves, _plane_of, _slug, _variance_fn,
                         layout_for_aspect_code, layout_size)
from mono_video import FRESH_SLOTS, HEAD_GROUPS, MONO_GROUPS
from mono_wire import MONO_PILOT_VALUES

PROFILE = 'stereo-slices'
MONO_PROFILE = 'mono-slices'        # one picture channel; the other is free
STATUS_MODE = tone_code.MONO_OFF
ENCODING_TYPE = v7.ENCODING_FILTER_CODES['nearest']
TABLES = Path(__file__).resolve().with_name('slice_tables.npz')
TABLES_SHA256 = '01881891d69d33a09ebc0777eed7459055ddf1ee4b64b246704b5a14f71953ea'
TABLE_FORMAT = 'v7-stereo-slices-1'
# Layout -> ((rows, cols) luma picture, (rows, cols) chroma picture); cols even.
SLICE_GRIDS = {
    '1:1': ((44, 44), (16, 16)), '4:3': ((38, 52), (14, 18)),
    '3:2': ((36, 56), (12, 20)), '16:9': ((34, 60), (10, 20)),
    '3:4': ((52, 38), (18, 14)), '2:3': ((56, 36), (20, 12)),
    '9:16': ((60, 34), (20, 10)),
}
MARKER_SLOTS = 8
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
CHROMA_RANK_WEIGHT = 4.0
SYNTHESIS_PICTURES = 400
SIDES = ('left', 'right')


def slice_shapes(layout):
    """((rows, cols) luma, chroma, chroma): the whole picture's rectangles."""
    layout_size(layout)
    luma, chroma = SLICE_GRIDS[layout]
    return (luma, chroma, chroma)


def split(picture):
    """The two checkerboard halves of ``picture`` as (rows, cols/2) arrays:
    the first holds the samples with row+column even."""
    rows, cols = picture.shape
    row = np.arange(rows)[:, None]
    odd = row % 2
    pair = 2*np.arange(cols//2)[None, :]
    return picture[row, pair+odd], picture[row, pair+1-odd]


def join(first, second):
    """Inverse of split()."""
    rows, half = first.shape
    picture = np.empty((rows, 2*half), dtype=np.result_type(first, second))
    row = np.arange(rows)[:, None]
    odd = row % 2
    pair = 2*np.arange(half)[None, :]
    picture[row, pair+odd] = first
    picture[row, pair+1-odd] = second
    return picture


def fill(half, first):
    """One checkerboard half -> the whole picture; each missing sample is the
    mean of its neighbours above, below, left and right (edges reflected)."""
    zero = np.zeros_like(half)
    picture = join(half, zero) if first else join(zero, half)
    padded = np.pad(picture, 1, mode='reflect')
    have = np.pad(join(np.ones_like(half), zero) if first else
                  join(zero, np.ones_like(half)), 1, mode='reflect')
    total = (padded[:-2, 1:-1]+padded[2:, 1:-1] +
             padded[1:-1, :-2]+padded[1:-1, 2:])
    count = (have[:-2, 1:-1]+have[2:, 1:-1]+have[1:-1, :-2]+have[1:-1, 2:])
    mask = have[1:-1, 1:-1] > 0
    return np.where(mask, picture, total/np.maximum(count, 1))


def widen(half):
    """A pair-averaged (rows, cols/2) picture on the whole (rows, cols) grid:
    the same cosine series, evaluated at twice the columns."""
    rows, cols = half.shape
    spectrum = np.zeros((rows, 2*cols))
    spectrum[:, :cols] = dctn(half, norm='ortho')
    return idctn(spectrum, norm='ortho')*np.sqrt(2.0)


def _corner(grid_values, shape):
    """Grid values -> the (rows, cols) picture with the same DCT corner."""
    rows, cols = shape
    spectrum = dctn(grid_values, norm='ortho')[:rows, :cols]
    return idctn(spectrum, norm='ortho')*np.sqrt(rows*cols/grid_values.size)


def _embed(picture, grid):
    """Inverse of _corner(): the picture as ``grid``-shaped values."""
    spectrum = np.zeros(grid)
    spectrum[:picture.shape[0], :picture.shape[1]] = dctn(picture, norm='ortho')
    return idctn(spectrum, norm='ortho')*np.sqrt(grid[0]*grid[1]/picture.size)


# ------------------------------------------------------------------- build
def slice_positions(layout):
    """(positions, marker): flattened V7_GRIDS indices of one channel's
    coefficients (luma half, Cb half, Cr half, marker), and the marker's
    indexes into that vector. A half's coefficient (u, v) is stored at grid
    position (u, v); the marker sits on luma positions outside the half."""
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    positions = []
    for plane, ((rows, cols), (grid_rows, grid_cols)) in enumerate(
            zip(slice_shapes(layout), v7.V7_GRIDS)):
        if cols % 2 or rows > grid_rows or cols//2 > grid_cols:
            raise ValueError(f'slice grid of {layout} does not fit plane {plane}')
        u, v = np.mgrid[:rows, :cols//2]
        positions.append(offsets[plane]+(u*grid_cols+v).ravel())
    (rows, cols), (grid_rows, grid_cols) = slice_shapes(layout)[0], v7.V7_GRIDS[0]
    spare = np.arange(MARKER_SLOTS)
    marker = (rows+spare//grid_cols)*grid_cols+spare % grid_cols
    if rows+MARKER_SLOTS//grid_cols >= grid_rows:
        raise ValueError(f'no room for the marker of {layout}')
    count = sum(len(part) for part in positions)
    positions.append(marker)
    positions = np.concatenate(positions).astype(np.int64)
    if len(positions) > FRESH_SLOTS:
        raise ValueError(f'slice grid of {layout} needs {len(positions)} slots')
    return positions, np.arange(count, count+MARKER_SLOTS)


def _half_variance(variance, layout, rng):
    """Per plane, the variance of a checkerboard half's grid coefficients:
    model pictures are drawn, split, and their halves' power averaged."""
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    out = []
    for plane, ((rows, cols), (grid_rows, grid_cols)) in enumerate(
            zip(slice_shapes(layout), v7.V7_GRIDS)):
        u, v = np.mgrid[:rows, :cols]
        whole = variance(offsets[plane]+(u*grid_cols+v).ravel()).reshape(rows, cols)
        dc = whole[0, 0]
        whole[0, 0] = 0.0
        # Grid coefficient -> the picture's own orthonormal coefficient.
        scale = rows*cols/(grid_rows*grid_cols)
        power = np.zeros((rows, cols//2))
        for _ in range(SYNTHESIS_PICTURES):
            picture = idctn(rng.standard_normal((rows, cols))*np.sqrt(whole*scale),
                            norm='ortho')
            first, second = split(picture)
            power += dctn(first, norm='ortho')**2+dctn(second, norm='ortho')**2
        power *= (grid_rows*grid_cols/(rows*cols//2))/(2*SYNTHESIS_PICTURES)
        power[0, 0] = dc
        out.append(power.ravel())
    return out


def _layout_tables(layout, curves, phase):
    variance = _variance_fn(curves, layout)
    positions, marker = slice_positions(layout)
    plane = _plane_of(positions)
    rng = np.random.default_rng(int(hashlib.sha256(layout.encode()).hexdigest()[:8], 16))
    lam = np.concatenate(_half_variance(variance, layout, rng) +
                         [np.zeros(MARKER_SLOTS)])
    luma = np.flatnonzero(plane == 0)[:-MARKER_SLOTS]
    # The marker is sent as strongly as a middling luma coefficient.
    lam[marker] = np.median(lam[luma])
    mu = np.zeros(len(positions))
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    for index, position in enumerate(positions):
        if position in offsets[:3]:
            mu[index] = curves[int(plane[index])][3]
    ranked = lam.copy()
    ranked[plane > 0] *= CHROMA_RANK_WEIGHT
    order = np.argsort(-ranked, kind='stable')
    gain = lam**-.25
    gain /= np.sqrt(np.mean(gain*gain*lam))
    tables = {'positions': positions, 'mu': mu, 'lam': lam, 'order': order,
              'gain': gain, 'marker': marker, 'unit_rms': np.float64(1.0)}
    probe = _assemble(tables, phase, 1.0)
    synth = (np.random.default_rng(v7.LEVEL_SEED).standard_normal(len(lam)) *
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


def layout_tables(layout):
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
    plane = _plane_of(np.asarray(tables['positions']))
    if template is None:
        return v7.Model(coder, mu, lam, order, gain, phase, scale, plane,
                        head, ranks, ENCODING_TYPE, priors,
                        mu32, lam32, gain32, phase32, priors32)
    return replace(template, coder=coder, mu=mu, lam=lam, order=order,
                   gain=gain, scale=scale, plane=plane, head=head,
                   rank_tables=ranks, encoding_type=ENCODING_TYPE,
                   block_prior_tables=priors, mu32=mu32, lam32=lam32,
                   gain32=gain32, block_prior_tables32=priors32)


class Half:
    """One decoded channel: its arrays and what the marker says it is."""

    __slots__ = ('kind', 'score', 'planes', 'layout', 'good', 'noise',
                 'damaged', 'steady')

    def __init__(self, kind, score, planes, layout, good=True, noise=0.0,
                 damaged=False, steady=None):
        self.kind, self.score, self.planes = kind, score, planes
        self.steady = score if steady is None else float(steady)
        self.layout, self.good, self.noise = layout, bool(good), float(noise)
        self.damaged = bool(damaged)


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
        self._tables = {}
        self._layout_of = {}
        self._marker = {}
        self._lock = threading.Lock()

    def reset(self):
        """Forget what each input channel was carrying."""
        self._marker.clear()

    def layout_for(self, aspect_code):
        if self.layout != 'auto':
            return self.layout
        return layout_for_aspect_code(aspect_code)

    def model_for(self, base_model, layout):
        """One channel's model for ``layout`` (either canonical base model)."""
        if id(base_model) in self._layout_of:
            return base_model                           # already a slice model
        key = (id(base_model), layout)
        with self._lock:
            if key not in self._models:
                name = v7.ENCODING_FILTERS[int(base_model.encoding_type)]
                canonical = float(v7._frozen_tables()[f'{name}/unit_rms'])
                tables = layout_tables(layout)
                model = _assemble(tables, base_model.phase,
                                  float(base_model.scale)*canonical,
                                  template=base_model)
                model._mono_wire_profile = self.wire_profile
                self._models[key] = model
                self._tables[id(model)] = tables
                self._layout_of[id(model)] = layout
            return self._models[key]

    def luma_sent_mask(self, layout):
        """The luma grid coefficients the whole picture carries."""
        rows, cols = slice_shapes(layout)[0]
        mask = np.zeros(v7.V7_GRIDS[0], bool)
        mask[:rows, :cols] = True
        return mask.ravel()

    def chroma_sent_masks(self, layout):
        """Per chroma grid: the coefficients the whole picture carries."""
        masks = []
        for (rows, cols), (grid_rows, grid_cols) in zip(
                slice_shapes(layout)[1:], v7.V7_GRIDS[1:]):
            mask = np.zeros((grid_rows, grid_cols), bool)
            mask[:rows, :cols] = True
            masks.append(mask.ravel())
        return masks

    # ---------------------------------------------------------------- sender
    def pictures(self, values, layout):
        """The whole picture's (luma, Cb, Cr) rectangles from grid values."""
        values = np.asarray(values, float)
        out, offset = [], 0
        for shape, (rows, cols) in zip(slice_shapes(layout), v7.V7_GRIDS):
            out.append(_corner(values[offset:offset+rows*cols].reshape(rows, cols),
                               shape))
            offset += rows*cols
        return out

    def channel_coefficients(self, model, pictures, kind):
        """One channel's coefficient vector; ``kind`` 'left', 'right', or
        'sum' (the pair average, with no marker: what mono playback of both
        channels would deliver, and what a single picture channel sends)."""
        tables = self._tables[id(model)]
        parts = []
        for picture, (grid_rows, grid_cols) in zip(pictures, v7.V7_GRIDS):
            first, second = split(picture)
            half = {'left': first, 'right': second}.get(kind)
            if half is None:
                half = (first+second)/2
            parts.append((dctn(half, norm='ortho') *
                          np.sqrt(grid_rows*grid_cols/half.size)).ravel())
        sign = {'left': 1.0, 'right': -1.0}.get(kind, 0.0)
        marker = np.asarray(tables['marker'])
        parts.append(sign*MARKER_LEVEL*MARKER_PATTERN *
                     np.sqrt(np.asarray(tables['lam'])[marker]))
        return np.concatenate(parts)

    def encode_packet(self, base_model, values, counter, aspect_code=0,
                      source_index=None, eof_marker=True, sides=SIDES):
        """One stereo packet: ``sides`` names what each output channel
        carries ('left', 'right', 'sum', or None for silence)."""
        if not eof_marker:
            raise ValueError('the stereo slices profile requires EOF markers')
        layout = self.layout_for(aspect_code)
        if layout is None:
            raise ValueError(f'no aspect layout for aspect code {aspect_code!r}')
        model = self.model_for(base_model, layout)
        pictures = self.pictures(values, layout)
        if source_index is None:
            source_index = int(counter)-1
        carried = [kind for kind in sides if kind is not None]
        if not carried or len(sides) != 2:
            raise ValueError('sides names what each of two channels carries')
        coefficients = [self.channel_coefficients(model, pictures, kind)
                        for kind in carried]
        # One packet build: with two channels each is its own mono wire.
        packet = v7.encode_pulse_frame_coeffs(
            model, coefficients[0], counter, aspect_code=aspect_code,
            source_index=source_index, pilot_tones=False, eof_marker=True,
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
               source_indices=None, eof_marker=True, sides=SIDES):
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
                               source_index=indexes[offset],
                               eof_marker=eof_marker, sides=sides)
            for offset, value in enumerate(values)])

    # -------------------------------------------------------------- receiver
    def half(self, model, result, channel=None):
        """The decoded channel of ``result`` as a Half. ``channel`` names the
        receiver's input channel, whose marker is then averaged over packets."""
        tables = self._tables[id(model)]
        layout = self._layout_of[id(model)]
        coeffs = np.asarray(result.coeffs, float)
        marker = np.asarray(tables['marker'])
        seen = coeffs[marker]
        equalized = result.diag.get('mono_fold_eq')
        if equalized is not None:
            # The equaliser's own output over its confidence: the receiver's
            # shrinking of noisy slots is undone, so noise does not read as
            # a sum.
            seen = (np.asarray(equalized[0])[marker] /
                    np.maximum(np.asarray(equalized[1])[marker], 1e-3))
        score = float(np.mean(
            seen/np.sqrt(np.asarray(tables['lam'])[marker]) *
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
        planes, offset = [], 0
        for (rows, cols), (grid_rows, grid_cols) in zip(
                slice_shapes(layout), v7.V7_GRIDS):
            count = rows*cols//2
            spectrum = coeffs[offset:offset+count].reshape(rows, cols//2)
            planes.append(idctn(spectrum, norm='ortho') *
                          np.sqrt(count/(grid_rows*grid_cols)))
            offset += count
        noise = result.diag.get('noise') or [0.0]
        return Half(kind, score, planes, layout, result.status != 'lost',
                    max(noise), damaged, steady)

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
            # The same half twice (one channel wired to both inputs).
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

    def values(self, halves):
        """Grid values for display from one or two decoded channels."""
        halves = self.choose(halves)
        layout = halves[0].layout
        kinds = [half.kind for half in halves]
        sums = [half for half in halves if half.kind == 'sum']
        leak = self.crosstalk(halves) if sorted(kinds) == sorted(SIDES) else 0.0
        pictures = []
        for plane in range(3):
            if sorted(kinds) == sorted(SIDES):
                by = {half.kind: half.planes[plane] for half in halves}
                left, right = by['left'], by['right']
                if leak:
                    left, right = ((left-leak*right)/(1-leak),
                                   (right-leak*left)/(1-leak))
                pictures.append(join(left, right))
            elif sums:
                # A mono sum (heard on one input or both): the pair average.
                pictures.append(widen(sum(half.planes[plane] for half in sums) /
                                      len(sums)))
            else:
                pictures.append(fill(halves[0].planes[plane],
                                     halves[0].kind == 'left'))
        return np.concatenate([_embed(picture, grid).ravel()
                               for picture, grid in zip(pictures, v7.V7_GRIDS)])


class SliceMonoWire:
    """The slices wire on one output channel, for the sender's mono-video
    plumbing: the picture channel carries the first checkerboard half and the
    other channel is left free (a soundtrack)."""

    status_mode = STATUS_MODE
    wire_profile = MONO_PROFILE
    requires_box = True
    disable_tail_memory = True
    fold_slots = 0

    def __init__(self, wire, side='right'):
        if side not in SIDES:
            raise ValueError("mono-video side must be 'left' or 'right'")
        self.wire, self.side, self.layout = wire, side, wire.layout
        self.carrier_index = SIDES.index(side)
        self._sides = ('left', None) if side == 'left' else (None, 'left')

    def layout_for(self, aspect_code):
        return self.wire.layout_for(aspect_code)

    def encode(self, model, values, start_counter=1, aspect_codes=None,
               source_indices=None, eof_marker=True):
        return self.wire.encode(model, values, start_counter, aspect_codes,
                                source_indices, eof_marker, sides=self._sides)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build',))
    build(parser.parse_args(argv))


if __name__ == '__main__':
    main()
