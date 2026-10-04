#!/usr/bin/env python3
"""Aspect-matched stereo Fold 500 (experimental ``aspect-fold-500`` profile).

The wire is V7's: the same packet framing, carriers, 2,320 slots per packet
(208 head, 2,016 body, 96 tail), metadata, pilots and EOF marker. What
changes is which 2,880 DCT coefficients those slots carry.

Fold 500 keeps the 48x40 luma / 24x20 chroma corners of the 96x80 / 48x40
sampling grids for every source, which suits a 5:6 picture: a 16:9 frame gets
40 horizontal steps across a width 1.8 times its height against 48 vertical
ones. This profile keeps, per plane, the coefficients nearest DC in
*picture* frequency for a chosen layout aspect W:H: coefficient (u, v) of a
plane lies at radius sqrt(u^2 + (v*H/W)^2), ordered by the integer key
u^2*W^2 + v^2*H^2 (ties by u, then v). The same count as today is kept
(1,920 / 480 / 480) and the next 500 luma coefficients are the fold guests.

Coefficient statistics are the canonical box model's, re-read as isotropic
in picture frequency: the canonical per-plane curve lam(rad) with
rad = hypot(u/rows, v/cols) (rows x cols the sampling grid, a 5:6 layout)
becomes rad = hypot(u/rows, v/(rows*W/H)); the DC mean and variance are the
canonical ones. The canonical curve is recovered exactly (to 3e-15) by the
same power law derive_tables() fits. Ranking, slot gains and the unit level
follow derive_tables(). All of it is frozen in aspect_tables.npz, whose
SHA-256 is pinned below, so both ends hold identical tables.

Signalling: coded status and pulse preamble ID 0 (the fold-off code, which
the live senders no longer emit). The tail mode is a receiver setting that
must match the sender. The layout is signalled: layout ``auto`` follows the
aspect code in each packet's metadata, and a sender with a fixed layout sends
that layout's code with the metadata screen bit (the picture boxed into it).

Tail modes (the 96 tail slots of each packet):
- chroma: V7's tail. The 656 lowest-ranked coefficients, all fine chroma,
  rotate 96 per packet over 7 packets; still pictures keep full 4:2:0
  chroma (480 per plane), moving pictures effectively about 304.
- split: 48 extra luma frequencies (beyond the fold guests) in every packet,
  48 chroma slots rotating (336 over 7 packets).
- luma: 96 extra luma frequencies in every packet; chroma is only what the
  head and body carry (~304 in all).
- fixed: the 96 strongest of the chroma tail's 656 coefficients in every
  packet, no rotation; the other 560 are not sent. Everything shown is from
  the current packet, so moving pictures have no stale colour (real modem,
  one new picture per packet, SSIMULACRA2 against chroma: +1 to +16; a held
  still on a clean channel loses 2-4). Derived from the frozen chroma tables.

    python test_modem_v7/aspect_fold.py build     # rebuild and print the pin
    python test_modem_v7/aspect_fold.py build-pixel   # the pixel grids' tables
"""
import argparse
import hashlib
import json
import threading
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from common import Grids, v7                                            # noqa: F401
from folding import FoldCodec, U_CLIP                                   # noqa: F401
import tone_code

TABLES = Path(__file__).resolve().with_name('aspect_tables.npz')
TABLES_SHA256 = '5ec85e57f14f12fffdf6379ae38a778872253ee50a93c3cb3b851d5b6962bb50'
PROFILE = 'aspect-fold-500'
STATUS_MODE = tone_code.FOLD_OFF
PULSE_PROFILE_CODE = STATUS_MODE
FOLD_SLOTS = 500
SIGNATURE_SLOTS = 16
TABLE_FORMAT = 'v7-aspect-fold-1'
# Tail mode -> luma slots fixed in every packet's 96 tail slots.
TAIL_LUMA_SLOTS = {'chroma': 0, 'split': 48, 'luma': 96, 'fixed': 0}
TAIL_MODES = tuple(TAIL_LUMA_SLOTS)
FROZEN_TAIL_MODES = ('chroma', 'split', 'luma')
# Tail mode -> tail slots that carry the same coefficients in every packet.
TAIL_FIXED_SLOTS = {'chroma': 0, 'split': 48, 'luma': 96, 'fixed': 96}
EXTRA_LUMA = max(TAIL_LUMA_SLOTS.values())
# (name, width, height). Names are the CLI/GUI values.
LAYOUTS = (('1:1', 1, 1), ('4:3', 4, 3), ('3:2', 3, 2), ('16:9', 16, 9),
           ('3:4', 3, 4), ('2:3', 2, 3), ('9:16', 9, 16))
LAYOUT_NAMES = tuple(name for name, _, _ in LAYOUTS)
LAYOUT_CHOICES = ('auto',) + LAYOUT_NAMES
# V7 metadata aspect code (V7_ASPECT_RATIOS) -> layout.
ASPECT_CODE_LAYOUT = ('1:1', '4:3', '3:2', '16:9', '1:1', '3:4', '2:3', '9:16')
PLANE_COUNTS = tuple(rows*cols for rows, cols in v7.V7_SHAPES)

# Pixel grids: per layout, a (rows, cols) luma rectangle whose coefficients
# the profile carries in full, so a picture of that many pixels arrives pixel
# for pixel (the sender's Pixel encode, the receiver's Pixel display). Chroma
# is half of it per axis; its 480 lowest picture frequencies per plane are
# sent.
#   robust: the rectangle fits the ordinary (host) slots, so it is as sturdy
#           as the hosts; nothing rides as a fold guest.
#   large:  the layout's exact aspect; its finest coefficients (384 to 496)
#           ride as fold guests, which only a clean link delivers exactly.
# A pixel grid is signalled by the metadata model bit (nearest), which the
# ordinary layouts never send; robust and large differ in fold signature.
PIXEL_GRIDS = {
    'robust': {'1:1': (42, 42), '4:3': (38, 50), '3:2': (36, 52),
               '16:9': (32, 58), '3:4': (50, 38), '2:3': (52, 36),
               '9:16': (58, 32)},
    'large': {'1:1': (48, 48), '4:3': (42, 56), '3:2': (40, 60),
              '16:9': (36, 64), '3:4': (56, 42), '2:3': (60, 40),
              '9:16': (64, 36)},
}
PIXEL_GRID_NAMES = tuple(PIXEL_GRIDS)
PIXEL_TABLES = Path(__file__).resolve().with_name('pixel_tables.npz')
PIXEL_TABLES_SHA256 = 'f5f1f8b76d52e653d477ba1fe2092aa3560ab6354f7a5d116a100e9a7b436f4f'
PIXEL_TABLE_FORMAT = 'v7-pixel-fold-1'
PIXEL_TAIL_MODES = ('chroma', 'fixed')
PIXEL_ENCODING_TYPE = v7.ENCODING_FILTER_CODES['nearest']


def pixel_shapes(layout, grid):
    """((rows, cols) luma, chroma, chroma) of a layout's pixel grid."""
    layout_size(layout)
    rows, cols = PIXEL_GRIDS[grid][layout]
    return ((rows, cols), (rows//2, cols//2), (rows//2, cols//2))


def pixel_positions(layout, grid):
    """(kept, guests): flattened V7_GRIDS indices of a layout's pixel grid.

    The fold's signature takes the last SIGNATURE_SLOTS hosts and their
    guests, which then carry no picture; those are positions outside the
    rectangle (finer than all of it), so every pixel coefficient is carried.
    kept: 1,920 luma (the rectangle's coefficients nearest DC in picture
    frequency, then outside filler), then per chroma plane the 480 nearest DC
    inside its rectangle (filler after a smaller rectangle); guests: the rest of the luma rectangle (none for
    'robust'), then SIGNATURE_SLOTS filler.
    """
    width, height = layout_size(layout)
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    kept, guests = [], None
    for plane, ((rows, cols), count, (rect_rows, rect_cols)) in enumerate(
            zip(v7.V7_GRIDS, PLANE_COUNTS, pixel_shapes(layout, grid))):
        u, v = (axis.ravel().astype(np.int64)
                for axis in np.mgrid[:rows, :cols])
        key = u*u*width*width + v*v*height*height
        inside = np.flatnonzero((u < rect_rows) & (v < rect_cols))
        if rect_rows > rows or rect_cols > cols:
            raise ValueError(f'pixel grid of {layout} does not fit plane {plane}')
        ordered = inside[np.lexsort((v[inside], u[inside], key[inside]))]
        filler = np.flatnonzero(key > key[inside].max())
        filler = filler[np.lexsort((v[filler], u[filler], key[filler]))]
        if plane:
            kept.append(offsets[plane]+np.concatenate(
                (ordered, filler))[:count])
            continue
        hosts = min(len(ordered), count-SIGNATURE_SLOTS)
        if grid == 'robust' and hosts < len(ordered):
            raise ValueError(f'robust pixel grid of {layout} exceeds the hosts')
        spare = count-hosts
        kept.append(np.concatenate((ordered[:hosts], filler[:spare])))
        if len(kept[0]) != count:
            raise ValueError(f'pixel grid of {layout} leaves no filler')
        guests = np.concatenate((ordered[hosts:],
                                 filler[spare:spare+SIGNATURE_SLOTS]))
    return np.concatenate(kept).astype(np.int64), guests.astype(np.int64)


def _slug(layout):
    return layout.replace(':', 'x')


def layout_size(layout):
    for name, width, height in LAYOUTS:
        if name == layout:
            return width, height
    raise ValueError(f'unknown aspect layout {layout!r}; choose {LAYOUT_NAMES}')


def layout_for_aspect_code(code):
    try:
        return ASPECT_CODE_LAYOUT[int(code)]
    except (TypeError, ValueError, IndexError):
        return None


def layout_positions(layout):
    """(kept, guests, extra): flattened V7_GRIDS indices in radius order.

    kept: 1,920 luma, 480 Cb, 480 Cr; guests: the next 500 luma (fold);
    extra: the 96 luma after those (tail luma slots).
    """
    width, height = layout_size(layout)
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    kept, guests, extra = [], None, None
    for plane, ((rows, cols), count) in enumerate(
            zip(v7.V7_GRIDS, PLANE_COUNTS)):
        u, v = (axis.ravel().astype(np.int64)
                for axis in np.mgrid[:rows, :cols])
        key = u*u*width*width + v*v*height*height
        ordered = np.lexsort((v, u, key))
        need = count + (FOLD_SLOTS+EXTRA_LUMA if plane == 0 else 0)
        chosen = ordered[:need]
        if need > rows*cols or key[chosen].max() >= min(
                key.reshape(rows, cols)[-1].min(),
                key.reshape(rows, cols)[:, -1].min()):
            raise ValueError(f'layout {layout} does not fit the {rows}x{cols} grid')
        kept.append(offsets[plane]+chosen[:count])
        if plane == 0:
            guests = offsets[0]+chosen[count:count+FOLD_SLOTS]
            extra = offsets[0]+chosen[count+FOLD_SLOTS:need]
    return (np.concatenate(kept).astype(np.int64), guests.astype(np.int64),
            extra.astype(np.int64))


# ------------------------------------------------------------------- build
def _canonical_curves():
    """Per-plane (a, k, q, dc_mu, dc_lam) of the canonical box model."""
    from scipy.optimize import curve_fit
    tables = v7._frozen_tables()
    mu, lam = tables['box/mu'], tables['box/lam']
    curves, offset = [], 0
    for (rows, cols), (grid_rows, grid_cols) in zip(v7.V7_SHAPES, v7.V7_GRIDS):
        plane = slice(offset, offset+rows*cols)
        rad = np.hypot(np.arange(rows)[:, None]/grid_rows,
                       np.arange(cols)[None, :]/grid_cols).ravel()
        mask = rad > 0
        law = lambda radius, a, k, q: a - q*np.log1p(k*radius)  # noqa: E731
        (a, k, q), _ = curve_fit(law, rad[mask], np.log(lam[plane][mask]),
                                 p0=(0, 20, 2), bounds=([-50, 0, 0], [50, 1e4, 10]),
                                 maxfev=20000)
        fitted = np.exp(law(rad[mask], a, k, q))
        if np.max(np.abs(fitted/lam[plane][mask]-1)) > 1e-9:
            raise ValueError('canonical variances do not follow the power law')
        curves.append((float(a), float(k), float(q), float(mu[plane][0]),
                       float(lam[plane][0])))
        offset += rows*cols
    return curves


def _variance_fn(curves, layout):
    width, height = layout_size(layout)
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])

    def variance(indexes):
        out = np.empty(len(indexes))
        for i, index in enumerate(indexes):
            plane = int(np.searchsorted(offsets, index, side='right')-1)
            rows, cols = v7.V7_GRIDS[plane]
            u, v = divmod(int(index-offsets[plane]), cols)
            a, k, q, _mu, dc = curves[plane]
            if u == 0 and v == 0:
                out[i] = dc
                continue
            radius = np.hypot(u/rows, v*height/(rows*width))
            out[i] = np.exp(a - q*np.log1p(k*radius))
        return out
    return variance


def _plane_of(positions):
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    return (np.searchsorted(offsets, positions, side='right')-1).astype(int)


def _mode_tables(layout, tail, curves, phase):
    """Coefficient vector, statistics and ranking for one layout and tail."""
    luma_slots = TAIL_LUMA_SLOTS[tail]
    variance = _variance_fn(curves, layout)
    kept, guests, extra = layout_positions(layout)
    luma = kept[:PLANE_COUNTS[0]]
    chroma = kept[PLANE_COUNTS[0]:]
    chroma_lam = variance(chroma)
    # Drop the weakest chroma to make room for the extra luma (count 2,880).
    if luma_slots:
        keep = np.sort(np.argsort(-chroma_lam, kind='stable')[:len(chroma)-luma_slots])
        chroma = chroma[keep]
    extra = extra[:luma_slots]
    positions = np.concatenate((luma, extra, chroma))
    lam = variance(positions)
    plane = _plane_of(positions)
    is_extra = np.zeros(len(positions), bool)
    is_extra[len(luma):len(luma)+len(extra)] = True
    # Head and body: the strongest 2,224 of the ordinary coefficients. Tail:
    # the extra luma first (fixed slots), then the rest by variance.
    ordinary = np.flatnonzero(~is_extra)
    ordinary = ordinary[np.argsort(-lam[ordinary], kind='stable')]
    order = np.concatenate((ordinary[:v7.BODY_END], np.flatnonzero(is_extra),
                            ordinary[v7.BODY_END:]))
    mu = np.zeros(len(positions))
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    for index, position in enumerate(positions):
        if position in offsets[:3]:
            mu[index] = curves[int(plane[index])][3]
    gain = lam**-.25
    gain /= np.sqrt(np.mean((gain*gain*lam)[order[:v7.BODY_END]]))
    tables = {'positions': positions, 'mu': mu, 'lam': lam, 'order': order,
              'gain': gain, 'guests': guests, 'guest_lam': variance(guests),
              'tail_luma_slots': np.int64(luma_slots),
              'unit_rms': np.float64(1.0)}
    probe_model = _assemble(tables, phase, 1.0)
    synth = (np.random.default_rng(v7.LEVEL_SEED).standard_normal(len(lam)) *
             np.sqrt(lam) + mu)
    probe = v7.encode_frame_coeffs(probe_model, synth, 1)
    tables['unit_rms'] = np.float64(np.sqrt(np.mean(probe**2)))
    return tables


def _pixel_tables(layout, grid, curves, phase):
    """Coefficient vector, statistics and ranking of one pixel grid (the
    chroma tail; the fixed tail is cut from it like the ordinary layouts')."""
    variance = _variance_fn(curves, layout)
    positions, guests = pixel_positions(layout, grid)
    lam = variance(positions)
    plane = _plane_of(positions)
    # The luma filler holds the fold signature, which must be sent as
    # strongly as a real host: give it the weakest real host's variance
    # (a hair less each, so it still ranks last).
    rect_rows, rect_cols = PIXEL_GRIDS[grid][layout]
    u, v = np.divmod(positions, v7.V7_GRIDS[0][1])
    filler = np.flatnonzero((plane == 0) & ((u >= rect_rows) | (v >= rect_cols)))
    real = np.flatnonzero((plane == 0) & (u < rect_rows) & (v < rect_cols))
    lam[filler] = lam[real].min()*(1-1e-6*np.arange(1, len(filler)+1))
    # Head and body: every luma coefficient (the fold hosts are body slots)
    # and the strongest chroma; tail: the remaining chroma by variance.
    luma = np.flatnonzero(plane == 0)
    chroma = np.flatnonzero(plane != 0)
    chroma = chroma[np.argsort(-lam[chroma], kind='stable')]
    body = np.concatenate((luma, chroma[:v7.BODY_END-len(luma)]))
    body = body[np.argsort(-lam[body], kind='stable')]
    order = np.concatenate((body, chroma[v7.BODY_END-len(luma):]))
    mu = np.zeros(len(positions))
    offsets = np.cumsum([0] + [rows*cols for rows, cols in v7.V7_GRIDS])
    for index, position in enumerate(positions):
        if position in offsets[:3]:
            mu[index] = curves[int(plane[index])][3]
    gain = lam**-.25
    gain /= np.sqrt(np.mean((gain*gain*lam)[order[:v7.BODY_END]]))
    tables = {'positions': positions, 'mu': mu, 'lam': lam, 'order': order,
              'gain': gain, 'guests': guests, 'guest_lam': variance(guests),
              'tail_luma_slots': np.int64(0), 'unit_rms': np.float64(1.0)}
    probe_model = _assemble(tables, phase, 1.0)
    synth = (np.random.default_rng(v7.LEVEL_SEED).standard_normal(len(lam)) *
             np.sqrt(lam) + mu)
    probe = v7.encode_frame_coeffs(probe_model, synth, 1)
    tables['unit_rms'] = np.float64(np.sqrt(np.mean(probe**2)))
    return tables


def build_pixel(_args=None):
    import io
    curves = _canonical_curves()
    phase = v7._frozen_tables()['phase']
    arrays = {}
    for grid in PIXEL_GRID_NAMES:
        for layout in LAYOUT_NAMES:
            for key, value in _pixel_tables(layout, grid, curves, phase).items():
                arrays[f'{grid}/{_slug(layout)}/chroma/{key}'] = np.asarray(value)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    blob = buffer.getvalue()
    PIXEL_TABLES.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    print(f"{PIXEL_TABLES.name}: {len(PIXEL_GRID_NAMES)} grids x {len(LAYOUT_NAMES)} layouts; pin PIXEL_TABLES_SHA256 = '{digest}'")
    return digest


def build(_args=None):
    import io
    curves = _canonical_curves()
    phase = v7._frozen_tables()['phase']
    arrays = {}
    for layout in LAYOUT_NAMES:
        for tail in FROZEN_TAIL_MODES:
            for key, value in _mode_tables(layout, tail, curves, phase).items():
                arrays[f'{_slug(layout)}/{tail}/{key}'] = np.asarray(value)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    blob = buffer.getvalue()
    TABLES.write_bytes(blob)
    digest = hashlib.sha256(blob).hexdigest()
    print(f"{TABLES.name}: {len(LAYOUT_NAMES)} layouts x {len(FROZEN_TAIL_MODES)} tails; pin TABLES_SHA256 = '{digest}'")
    return digest


# ------------------------------------------------------------------- model
@lru_cache(maxsize=2)
def _frozen(pixel=False):
    path, pin = ((PIXEL_TABLES, PIXEL_TABLES_SHA256) if pixel else
                 (TABLES, TABLES_SHA256))
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != pin:
        raise ValueError(f'{path.name} SHA-256 {digest[:12]}… does not match the '
                         f'pinned {pin[:12]}…')
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _fixed_tail_tables(tables):
    """The chroma-tail tables cut to what one packet carries: head, body and
    the 96 strongest tail coefficients, which then ride in every packet."""
    order = np.asarray(tables['order'])
    sent = order[:v7.BODY_END+v7.TAIL_PER]
    keep = np.sort(sent)
    index = np.full(len(order), -1, dtype=np.int64)
    index[keep] = np.arange(len(keep))
    out = {key: np.asarray(tables[key])[keep]
           for key in ('positions', 'mu', 'lam', 'gain')}
    out.update(order=index[sent], guests=tables['guests'],
               guest_lam=tables['guest_lam'],
               tail_luma_slots=np.int64(v7.TAIL_PER),
               unit_rms=tables['unit_rms'])
    return out


def layout_tables(layout, tail, pixel=None):
    """One layout's tables; ``pixel`` names a pixel grid (PIXEL_GRIDS)."""
    if pixel and pixel not in PIXEL_GRIDS:
        raise ValueError(f'unknown pixel grid {pixel!r}; choose {PIXEL_GRID_NAMES}')
    if pixel and tail not in PIXEL_TAIL_MODES:
        raise ValueError(f'pixel grids have no {tail!r} tail; choose {PIXEL_TAIL_MODES}')
    if tail == 'fixed':
        return _fixed_tail_tables(layout_tables(layout, 'chroma', pixel))
    frozen = _frozen(bool(pixel))
    prefix = (f'{pixel}/' if pixel else '')+f'{_slug(layout)}/{tail}/'
    tables = {key[len(prefix):]: value for key, value in frozen.items()
              if key.startswith(prefix)}
    if not tables:
        raise ValueError(f'no frozen tables for layout {layout!r}, tail {tail!r}')
    return tables


class AspectCoder:
    """V7 coder interface over an arbitrary set of sampling-grid positions."""

    def __init__(self, positions):
        self.shapes = list(v7.V7_SHAPES)
        self.grids = list(v7.V7_GRIDS)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.count = len(self.positions)
        self.source_count = int(sum(rows*cols for rows, cols in self.grids))
        self.truncated = True
        self.gains = np.ones(self.count)
        self.variance = None
        self._grid = Grids(self.grids)

    def forward(self, values):
        return self._grid.forward(np.asarray(values, float))[self.positions]

    def inverse(self, sent, reliability=None, noise_variance=None):
        if reliability is not None:
            raise NotImplementedError('aspect layouts decode without the Wiener path')
        full = np.zeros(self.source_count)
        full[self.positions] = np.asarray(sent, float)
        return self._grid.inverse(full)


def _rank_tables(order, luma_slots):
    """Per tail phase: head/body as V7, then the tail slots' contents.

    The first ``luma_slots`` tail ranks are in every packet; the remaining
    tail slots rotate through the rest over the 7 tail phases.
    """
    order = np.asarray(order)
    head_body = order[:v7.BODY_END]
    fixed = order[v7.BODY_END:v7.BODY_END+luma_slots]
    pool = order[v7.BODY_END+luma_slots:]
    rotating = v7.TAIL_PER-luma_slots
    tables = []
    for phase in range(v7.TAIL_PHASES):
        part = pool[rotating*phase:rotating*(phase+1)] if rotating else pool[:0]
        tail = np.concatenate((fixed, part))
        tail = np.pad(tail, (0, v7.TAIL_PER-len(tail)), constant_values=-1)
        # frame_ranks(order, 0) takes phase 0's 96 tail entries from index
        # BODY_END, which is exactly this phase's tail.
        tables.append(v7.frame_ranks(np.concatenate((head_body, tail)), 0))
    return tuple(tables)


def _assemble(tables, phase, target_rms, template=None):
    order = np.asarray(tables['order'])
    lam = np.asarray(tables['lam'])
    gain = np.asarray(tables['gain'])
    mu = np.asarray(tables['mu'])
    head = np.zeros(len(lam), bool)
    head[order[:v7.HEAD]] = True
    ranks = _rank_tables(order, int(tables['tail_luma_slots']))
    priors = tuple(v7.block_priors(gain, lam, idx) for idx in ranks)
    mu32 = np.asarray(mu, dtype=np.float32)
    lam32 = np.asarray(lam, dtype=np.float32)
    gain32 = np.asarray(gain, dtype=np.float32)
    phase32 = np.asarray(phase, dtype=np.complex64)
    priors32 = tuple(np.asarray(table, dtype=np.float32) for table in priors)
    for array in (mu32, lam32, gain32, phase32, *priors32):
        array.setflags(write=False)
    coder = AspectCoder(tables['positions'])
    scale = target_rms/float(tables['unit_rms'])
    plane = _plane_of(np.asarray(tables['positions']))
    if template is None:
        return v7.Model(coder, mu, lam, order, gain, phase, scale, plane,
                        head, ranks, v7.ENCODING_FILTER_CODES['box'], priors,
                        mu32, lam32, gain32, phase32, priors32)
    return replace(template, coder=coder, mu=mu, lam=lam, order=order,
                   gain=gain, scale=scale, plane=plane, head=head, rank_tables=ranks,
                   block_prior_tables=priors, mu32=mu32, lam32=lam32,
                   gain32=gain32, block_prior_tables32=priors32)


class AspectFoldCodec(FoldCodec):
    """Fold 500's host/guest split over an aspect layout's coefficients."""

    def __init__(self, model, layout, tail, tables, step, pixel=None):
        self.model, self.M = model, len(tables['guests'])
        self.layout, self.tail, self.pixel = layout, tail, pixel
        self.filter, self.conf_min = 'box', .9
        self.design_db, self.fitted_on = 30.0, f'aspect layout {layout} (analytic)'
        self.signature, self.noise_max = SIGNATURE_SLOTS, .3
        self.grid = Grids(v7.V7_GRIDS)
        self.kept = np.asarray(tables['positions'], dtype=np.int64)
        rank = np.empty(len(model.order), int)
        rank[model.order] = np.arange(len(model.order))
        body = np.flatnonzero((np.asarray(model.plane) == 0) &
                              (rank >= v7.HEAD) & (rank < v7.BODY_END))
        body = body[np.argsort(rank[body])]
        self.hosts = body[-self.M:]                  # weakest luma body slots
        self.guests = np.asarray(tables['guests'], dtype=np.int64)
        self.sd_host = np.sqrt(model.lam[self.hosts])
        self.sd_guest = np.sqrt(np.asarray(tables['guest_lam'], float))
        self.train = []
        self.set_step(step)
        if ASPECT_COMPAND is not None:
            self.use_compand(ASPECT_COMPAND['limit'], ASPECT_COMPAND['mu'],
                             ASPECT_COMPAND['step'],
                             ASPECT_COMPAND['guest_noise_max'])
        self._set_identity()
        if self.pixel:
            # The two grids of a layout share their signature slots, so
            # their patterns must not be alike by chance: one pattern per
            # layout, with every other sign flipped for each later grid
            # (exactly uncorrelated, so the wrong grid scores zero).
            seed = int(hashlib.sha256(
                f'{PIXEL_TABLES_SHA256}/{layout}/{tail}'.encode()).hexdigest()[:16], 16)
            self.pattern = np.random.default_rng(seed).choice(
                [-1.0, 1.0], self.signature)
            flip = PIXEL_GRID_NAMES.index(self.pixel)
            if flip:
                self.pattern[np.arange(self.signature) % (2*flip) >= flip] *= -1

    def table(self):
        table = super().table()
        table.update({'format': TABLE_FORMAT, 'layout': self.layout,
                      'tail': self.tail, 'aspect_tables_sha256': TABLES_SHA256})
        if self.pixel:
            table.update({'format': PIXEL_TABLE_FORMAT, 'pixel': self.pixel,
                          'aspect_tables_sha256': PIXEL_TABLES_SHA256})
        return table


# Companded guests for the aspect profile (see folding.py): step, guest
# limit (model standard deviations), mu, and the symbol noise past which the
# guests are dropped. None restores the linear guests of Fold 500.
ASPECT_COMPAND = {'step': 1.0, 'limit': 12.0, 'mu': 4.0, 'guest_noise_max': .1}


@lru_cache(maxsize=1)
def _fold500_step():
    path = Path(__file__).resolve().with_name('fold_table_500.json')
    return float(json.loads(path.read_text())['D'])


class AspectFoldWire:
    """Sender/receiver glue for the aspect-matched stereo Fold 500 profile."""

    status_mode = STATUS_MODE
    pulse_profile_code = PULSE_PROFILE_CODE
    fold_slots = FOLD_SLOTS
    wire_profile = PROFILE

    def __init__(self, layout='auto', tail='chroma', pixel=None):
        if layout not in LAYOUT_CHOICES:
            raise ValueError(f'unknown aspect layout {layout!r}; choose {LAYOUT_CHOICES}')
        if tail not in TAIL_MODES:
            raise ValueError(f'unknown aspect tail mode {tail!r}; choose {TAIL_MODES}')
        if pixel and pixel not in PIXEL_GRIDS:
            raise ValueError(f'unknown pixel grid {pixel!r}; choose {PIXEL_GRID_NAMES}')
        if pixel and tail not in PIXEL_TAIL_MODES:
            raise ValueError(f'pixel grids have no {tail!r} tail; choose {PIXEL_TAIL_MODES}')
        self.layout, self.tail, self.pixel = layout, tail, pixel or None
        self._inside = {}
        self._models = {}
        self._codecs = {}
        self._tails = {}
        self._lock = threading.Lock()

    def layout_for(self, aspect_code):
        """The layout this end uses for a packet with this aspect code."""
        if self.layout != 'auto':
            return self.layout
        return layout_for_aspect_code(aspect_code)

    def model_for(self, base_model, layout):
        # Pixel grids ride under the nearest model's metadata bit, so the
        # receiver's base model for them may be either canonical model.
        if (not self.pixel and
                int(base_model.encoding_type) != v7.ENCODING_FILTER_CODES['box']):
            raise ValueError('aspect-fold-500 requires the box model')
        key = (id(base_model), layout)
        with self._lock:
            if key not in self._models:
                name = v7.ENCODING_FILTERS[int(base_model.encoding_type)]
                canonical = float(v7._frozen_tables()[f'{name}/unit_rms'])
                target = float(base_model.scale)*canonical
                tables = layout_tables(layout, self.tail, self.pixel)
                model = _assemble(tables, base_model.phase, target,
                                  template=base_model)
                if self.pixel:
                    model = replace(model, encoding_type=PIXEL_ENCODING_TYPE)
                self._models[key] = model
                self._codecs[id(model)] = AspectFoldCodec(
                    model, layout, self.tail, tables, _fold500_step(),
                    pixel=self.pixel)
            return self._models[key]

    def pixel_shapes(self, layout):
        """The layout's pixel grid shapes, or None off a pixel grid."""
        return pixel_shapes(layout, self.pixel) if self.pixel else None

    def _outside(self, layout):
        """Grid positions outside the layout's pixel rectangles."""
        if layout not in self._inside:
            masks = []
            for (rows, cols), (rect_rows, rect_cols) in zip(
                    v7.V7_GRIDS, pixel_shapes(layout, self.pixel)):
                mask = np.ones((rows, cols), bool)
                mask[:rect_rows, :rect_cols] = False
                masks.append(mask.ravel())
            self._inside[layout] = np.flatnonzero(np.concatenate(masks))
        return self._inside[layout]

    def codec(self, model):
        return self._codecs[id(model)]

    # ---------------------------------------------------------------- sender
    def encode_coefficients(self, base_model, values, aspect_code):
        """(model, folded coefficients) for one frame's grid values."""
        layout = self.layout_for(aspect_code)
        if layout is None:
            raise ValueError(f'no aspect layout for aspect code {aspect_code!r}')
        model = self.model_for(base_model, layout)
        return model, self.codec(model).encode_coefficients(values)

    # -------------------------------------------------------------- receiver
    @property
    def rotates(self):
        return TAIL_FIXED_SLOTS[self.tail] < v7.TAIL_PER

    def tail_prior(self, model, layout):
        """Previous-packet coefficients for the rotating tail (own store)."""
        if not self.rotates:
            return model.mu
        store = self._tails.get(layout)
        if store is None:
            # One store at a time: a layout change is a discontinuity.
            self._tails.clear()
            store = self._tails[layout] = v7.TailStore()
        return store.prior(model)

    def remember(self, model, layout, result):
        if not self.rotates or result.status == 'lost':
            return
        tail_slice = result.diag.get('tail_slice')
        store = self._tails.get(layout)
        if tail_slice is not None and store is not None:
            store.update(model, result.coeffs, tail_slice)

    def reset(self):
        self._tails.clear()

    def values(self, model, result, metadata_confirmed=True):
        """Grid values for display: unfold the hosts, place the layout."""
        codec = self.codec(model)
        eq = result.diag.get('fold_eq')
        if eq is None:
            full = codec.plain(result.coeffs)
        else:
            coeffs, xhat, conf = result.coeffs, eq[0], eq[1]
            fusion = result.diag.get('temporal_fusion')
            if fusion is not None:
                from animation_modem.v7_temporal import fused_packet
                coeffs, xhat, conf = fused_packet(
                    fusion, model, coeffs, xhat, conf,
                    key=('aspect', codec.layout, self.pixel,
                         result.diag.get('direction')))
                result.diag['fusion_gain'] = fusion.last_gain
            full = codec.decode(coeffs, xhat, conf, fallback=True,
                                metadata_confirmed=metadata_confirmed)
            result.diag['aspect_signature_score'] = codec.last_score
        if self.pixel:
            # Filler slots carry no picture: only the rectangle is shown.
            full[self._outside(codec.layout)] = 0.0
            result.diag['pixel_shapes'] = pixel_shapes(codec.layout, self.pixel)
        return codec.grid.inverse(full)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'build-pixel'))
    args = parser.parse_args(argv)
    (build_pixel if args.command == 'build-pixel' else build)(args)


if __name__ == '__main__':
    main()
