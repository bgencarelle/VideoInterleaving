"""Experimental source coding over the unchanged production V7 aspect wire.

Only benchmark-local models/hooks are installed. Never used by live defaults.
The receiver is explicitly configured for the agreed experimental profile;
packet timing, metadata, status and EOF still come from received audio.
"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import time

import numpy as np
from scipy.fft import dctn, idctn

from animation_modem import v7, v7_kernels
from animation_modem.v7_source_dct import (
    _block_means_f64, _decimate, _direct_plan, _separable, KernelFrame)
from tools import v7_live

v7_live._ensure_test_modem_path()
import aspect_fold
from aspect_fold import AspectFoldWire, AspectFoldCodec, _assemble
from aspect_mono import AspectMonoWire, AspectMonoCodec
from mono_video import fresh_rank_tables, colour_order, MonoFreshFoldWire
import tone_code


TARGET = .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10))
SPLITS = {'current': (1920, 480, 480), 'detail': (2160, 360, 360),
          'more-detail': (2400, 240, 240), 'max-detail': (2848, 16, 16)}
OFFSETS = np.cumsum([0]+[r*c for r, c in v7.V7_GRIDS])


def split_planes(values):
    return [np.asarray(values)[OFFSETS[i]:OFFSETS[i+1]].reshape(shape)
            for i, shape in enumerate(v7.V7_GRIDS)]


def full_coefficients(values):
    return np.concatenate([dctn(p, norm='ortho').ravel() for p in split_planes(values)])


def positions_for(layout, split):
    width, height = aspect_fold.layout_size(layout)
    kept, guests = [], None
    for i, ((rows, cols), count) in enumerate(zip(v7.V7_GRIDS, SPLITS[split])):
        u, vv = np.mgrid[:rows, :cols]
        order = np.lexsort((vv.ravel(), u.ravel(),
                           (u*u*width*width+vv*vv*height*height).ravel()))
        kept.append(OFFSETS[i]+order[:count])
        if i == 0:
            guests = order[count:count+500]
    return np.concatenate(kept), guests


def masks_for(codec):
    mask = np.zeros(OFFSETS[-1], bool)
    kept = codec.kept
    if codec.sent_model_indices is not None:
        kept = kept[codec.sent_model_indices]
    mask[kept] = True
    mask[codec.guests[:-codec.signature]] = True
    mask[codec.kept[codec.hosts[-codec.signature:]]] = False
    return split_planes(mask)


def source_values(rgb, transform, kernel=None, masks=None, profile=None,
                  repair=True):
    """Convert native pixels, then reuse V7's block/decimation/DCT geometry.

    Nonlinear color conversions happen BEFORE averaging, unlike an invalid
    conversion of already averaged RGB. Normalized planes have fixed ranges.
    Brightness repair uses the RGB inverse gradient, not an assumed PCA luma.
    """
    x = np.asarray(rgb, float)/255
    height, width = x.shape[:2]
    planes = transform.forward(x)
    prefilter = (kernel.kernel.prefilter(kernel.params) or {}
                 if kernel is not None and kernel.kernel.has_prefilter else {})
    full = prefilter.get('full_source', False)
    factor = prefilter.get('preshrink', 4)
    by = 1 if full else max(1, int(height/(factor*96)+.5))
    bx = 1 if full else max(1, int(width/(factor*80)+.5))
    rows, cols = -(-height//by), -(-width//bx)
    dy = not full and height % (2*by) == 0 and rows//2 >= 1.75*96
    dx = not full and width % (2*bx) == 0 and cols//2 >= 1.75*80
    means = _block_means_f64(np.ascontiguousarray(planes), by, bx, rows, cols)
    means = [_decimate(p, dy, dx) if dy or dx else p for p in means]
    blocks = [(by, bx, int(dy), int(dx))]*3
    ay = dy and height % (4*by) == 0 and means[0].shape[0]//2 >= 1.75*48
    ax = dx and width % (4*bx) == 0 and means[0].shape[1]//2 >= 1.75*40
    if ay or ax:
        means[1:] = [_decimate(p, ay, ax) for p in means[1:]]
    blocks[1:] = [(by, bx, int(dy)+int(ay), int(dx)+int(ax))]*2
    frame = (KernelFrame(kernel, v7.V7_GRIDS, v7.V7_SHAPES, masks, means)
             if kernel is not None else None)
    result = []
    for i, (plane, shape, block) in enumerate(zip(means, v7.V7_GRIDS, blocks)):
        left, right = _direct_plan(height, width, *shape, *block)[-2:]
        grid = _separable(left, np.ascontiguousarray(plane), right)
        gain = frame.gain(i) if frame is not None else None
        if gain is not None:
            grid = idctn(dctn(grid, norm='ortho')*gain, norm='ortho')
        result.append(grid)
    values = np.concatenate([p.ravel()*2-1 for p in result])
    if repair and masks is not None and transform.base not in ('gray', 'hsv', 'hsl'):
        values = repair_brightness(values, x, transform, masks)
    if frame is not None:
        values = frame.post(values, v7.V7_GRIDS)
        error = kernel.kernel.take_error()
        if error:
            raise RuntimeError(f'kernel bypass invalidates benchmark: {error}')
    # Avoid giving alternatives unbounded source values unavailable in production.
    return np.clip(values, -1, 1)


def _resize_float(plane, size):
    from PIL import Image
    return np.asarray(Image.fromarray(np.float32(plane), 'F').resize(
        size, Image.Resampling.BILINEAR), float)


def repair_brightness(values, source, transform, masks):
    """Limited correction along the actual RGB brightness gradient.

    Corrections are projected to each component's carried band. This is an
    explicitly experimental generalization, not the production YCbCr repair.
    """
    from tools.v7_color_transforms import linearize
    planes = [(p+1)/2 for p in split_planes(values)]
    target = _resize_float(linearize(source)@np.array([.2126, .7152, .0722]), (80, 96))
    weights = np.array([.2126, .7152, .0722])
    for _ in range(3):
        shown = np.stack([_resize_float(p, (80, 96)) for p in planes], -1)
        rgb = transform.inverse(shown)
        light = linearize(np.clip(rgb, 0, 1))@weights
        derivatives = []
        for i in range(3):
            perturbed = shown.copy()
            perturbed[..., i] += .001
            derivatives.append((linearize(np.clip(transform.inverse(perturbed),
                                                  0, 1))@weights-light)/.001)
        gradient = np.stack(derivatives, -1)
        step = (target-light)/np.maximum((gradient*gradient).sum(-1), .01)
        for i, plane in enumerate(planes):
            correction = _resize_float(np.clip(step*gradient[..., i], -.05, .05),
                                       plane.shape[::-1])
            correction = idctn(dctn(correction, norm='ortho')*masks[i], norm='ortho')
            planes[i] = plane+correction
    return np.concatenate([p.ravel()*2-1 for p in planes])


def fit_statistics(frames, transform):
    total = np.zeros(OFFSETS[-1])
    squares = np.zeros_like(total)
    count = 0
    for rgb in frames:
        c = full_coefficients(source_values(rgb, transform, repair=False))
        total += c
        squares += c*c
        count += 1
    if count < 2:
        raise ValueError('at least two training images required')
    mean = total/count
    variance = np.maximum((squares-total*total/count)/(count-1), 0)
    floor = np.concatenate([1e-5/(1+(np.mgrid[:r, :c]**2).sum(0)).ravel()
                            for r, c in v7.V7_GRIDS])
    return mean, np.maximum(variance, floor)


def fitted_tables(base, layout, split, mean, variance):
    positions, guests = positions_for(layout, split)
    lam = variance[positions]
    order = np.argsort(-lam, kind='stable')
    # Ensure every component's DC is sent. Reserve enough plane-zero body
    # coefficients for the 500-host fold, also with unusual color statistics.
    plane = aspect_fold._plane_of(positions)
    dc = np.flatnonzero(np.isin(positions, OFFSETS[:3]))
    front = np.concatenate((dc, order[~np.isin(order, dc)]))
    luma = front[plane[front] == 0]
    protected = np.unique(np.concatenate((front[:v7.HEAD], luma[:1000])))
    body = front[np.isin(front, protected)]
    rest = front[~np.isin(front, body)]
    order = np.concatenate((body, rest))
    gain = lam**-.25
    gain /= np.sqrt(np.mean((gain*gain*lam)[order[:v7.BODY_END]]))
    tables = {'positions': positions, 'guests': guests,
              'mu': mean[positions], 'lam': lam, 'order': order, 'gain': gain,
              'guest_lam': variance[guests], 'unit_rms': np.float64(1)}
    probe = _assemble(tables, base.phase, 1)
    synth = np.random.default_rng(v7.LEVEL_SEED).standard_normal(2880)*np.sqrt(lam)+tables['mu']
    audio = v7.encode_frame_coeffs(probe, synth, 1)
    tables['unit_rms'] = np.float64(np.sqrt(np.mean(audio*audio)))
    return tables


class FittedAspectWire(AspectFoldWire):
    def __init__(self, base, layout, tables):
        super().__init__(layout)
        model = _assemble(tables, base.phase, TARGET, template=base)
        self.fitted_model = model
        self._codecs[id(model)] = AspectFoldCodec(model, layout, tables, 1)

    def model_for(self, base_model, layout):
        if layout != self.layout:
            raise ValueError('experimental model is frozen to one layout')
        return self.fitted_model


class FittedMonoWire(AspectMonoWire):
    def __init__(self, aspect, base):
        self.layout = aspect.layout
        model = aspect.fitted_model
        ranks = fresh_rank_tables(model, colour_order(model))
        priors = tuple(v7.block_priors(model.gain, model.lam, rank) for rank in ranks)
        self.fitted_model = replace(
            model, rank_tables=ranks, block_prior_tables=priors,
            block_prior_tables32=tuple(np.asarray(p, np.float32) for p in priors),
            scale=float(model.scale*np.sqrt(2)))
        self.fitted_model._mono_wire_profile = self.wire_profile
        codec = AspectMonoCodec(model, self.layout)
        if len(codec.hosts) != 500 or len(codec.guests) != 500:
            raise ValueError('allocation cannot supply 500 mono fold hosts/guests')
        self.fitted_codec = codec
        MonoFreshFoldWire.__init__(self, base, side='left')

    def model_for(self, model, layout=None):
        if layout is not None and layout != self.layout:
            raise ValueError('experimental model is frozen to one layout')
        self._codec_by_model_id[id(self.fitted_model)] = self.fitted_codec
        return self.fitted_model


class ColorWire:
    def __init__(self, profile, layout, transform, kernel='viewer_solve',
                 split='current', statistics=None, kernel_params=None, tables=None):
        self.profile, self.layout, self.transform = profile, layout, transform
        self.base = v7.load_model(TARGET, 'box')
        self.selection = v7_kernels.open_registry().select(
            kernel, kernel_params, profile=profile)
        self.kernel_name, self.split = kernel, split
        self.production = transform.name == 'pillow-ycbcr' and statistics is None and tables is None
        if self.production:
            self.wire = (AspectFoldWire(layout) if profile == 'aspect-fold-500'
                         else AspectMonoWire(self.base, side='left', layout=layout))
        else:
            tables = tables or fitted_tables(self.base, layout, split, *statistics)
            aspect = FittedAspectWire(self.base, layout, tables)
            self.wire = aspect if profile == 'aspect-fold-500' else FittedMonoWire(aspect, self.base)
        self.tables = tables
        self.model = self.wire.model_for(self.base, layout)
        self.codec = (self.wire.codec(self.model) if profile == 'aspect-fold-500'
                      else self.wire._codec(self.model))
        self.masks = masks_for(self.codec)
        if self.production:
            # Match exactly the live sender's mask contract, including its
            # signature/filler positions in the luma support.
            self.masks = [self.codec.sent_luma_mask().reshape(v7.V7_GRIDS[0])]
            self.masks += [mask.reshape(shape) for mask, shape in zip(
                v7_live._chroma_sent_masks(self.codec), v7.V7_GRIDS[1:])]
        self.aspect = v7.V7_ASPECT_NAMES.index(layout)

    def record(self):
        model = self.model
        digest = hashlib.sha256()
        for x in (model.mu, model.lam, model.gain, model.coder.positions,
                  *model.rank_tables):
            digest.update(np.asarray(x).tobytes())
        return {'profile': self.profile, 'layout': self.layout,
                'transform': self.transform.record(), 'kernel': self.kernel_name,
                'kernel_params': self.selection.params if self.selection else {},
                'split': self.split, 'production': self.production,
                'model_digest': digest.hexdigest(), 'fold': self.codec.table(),
                'positions': model.coder.positions.tolist(),
                'target': TARGET, 'model_scale': model.scale,
                'slots': 2320 if self.profile == 'aspect-fold-500' else 1264,
                'receiver_profile_selection': 'preconfigured; received status verified',
                'brightness_repair': 'production' if self.production else 'inverse-RGB-gradient'}

    def values(self, rgb):
        if not self.production:
            return source_values(rgb, self.transform, self.selection, self.masks,
                                 self.profile)
        return v7_live._values(
            self.base, rgb, 'box', brightness=1.0, gamma=1.0, dct_encode=True,
            dct_options={'kernel': self.selection},
            chroma_sent_for=lambda _: self.masks[1:],
            kernel_masks_for=lambda _: self.masks)[0]

    def encode_packet(self, rgb, counter, source_index):
        values = self.values(rgb)
        if self.profile == 'aspect-mono-500':
            return self.wire.encode_packet(self.base, values, counter,
                                           self.aspect, source_index)
        model, coefficients = self.wire.encode_coefficients(self.base, values, self.aspect)
        packet = v7_live._encode_pulse_frame_coeffs(
            model, coefficients, counter, aspect_code=self.aspect,
            source_index=source_index,
            pulse_profile_code=self.wire.pulse_profile_code)
        return v7_live._add_coded_pilots(packet, counter, 500, mode=self.wire.status_mode)

    @contextmanager
    def receiver(self):
        decoder = v7_live._AdaptiveProfileDecoder(
            v7_live._experimental_fold(500), self.base,
            aspect_layout=self.layout)
        if self.profile == 'aspect-fold-500':
            decoder.aspect_wire = self.wire
            mode = decoder.aspect_mode
        else:
            decoder.aspect_mono_wire = self.wire
            mode = decoder.aspect_mono_mode
            decoder.mono_wires[mode] = self.wire
        decoder.active_mode = decoder.dispatch_mode = mode
        decoder.bind_state(v7.PulseState(tail_memory=False))
        decoder.install()
        try:
            with tone_code.coded_pilot_timing():
                yield decoder
        finally:
            decoder.uninstall()

    def decode(self, audio, rate=96000):
        started = time.perf_counter()
        with self.receiver() as receiver:
            results, info = v7.decode_pulse_stream(
                self.base, audio, models={self.base.encoding_type: self.base},
                state=receiver.state, sample_rate=rate,
                pilot_timing='tone-seeded')
            pictures = []
            for result in results:
                if result.status != 'lost' and result.diag.get('displayable', True):
                    values = receiver.values(self.base, result)
                    pictures.append((result, values))
                else:
                    pictures.append((result, None))
        return pictures, info, time.perf_counter()-started

    def reconstruct(self, values, factor=1):
        planes = []
        for plane in split_planes(values):
            if factor != 1:
                coefficients = dctn(plane, norm='ortho')
                padded = np.zeros(np.array(plane.shape)*factor)
                padded[:plane.shape[0], :plane.shape[1]] = coefficients*factor
                plane = idctn(padded, norm='ortho')
            planes.append((plane+1)/2)
        size = planes[0].shape[::-1]
        rgb = self.transform.inverse(np.stack([_resize_float(p, size) for p in planes], -1))
        return rgb
