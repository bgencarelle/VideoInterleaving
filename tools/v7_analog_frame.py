"""Matched, frame-independent analog picture coding on the existing V7 wire."""
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn
from PIL import Image

from animation_modem import v7, v7_kernels
from animation_modem.v7_source_dct import (
    _block_means_f64, _decimate, _direct_plan, _separable, KernelFrame)
from tools.v7_color_wire import ColorWire, TARGET, SPLITS
from tools.v7_color_transforms import ColorTransform, linearize
from tools import v7_live

v7_live._ensure_test_modem_path()
from common import Grids
from folding import FoldCodec
from aspect_fold import AspectCoder, AspectFoldWire, _assemble, layout_size
from mono_video import MonoFreshFoldWire, fresh_rank_tables
from aspect_mono import AspectMonoWire


GRID_FAMILIES = {
    'legacy': ((96, 80), (48, 40), (48, 40)),
    'equal': ((96, 80),)*3,
    'expanded': ((144, 120),)*3,
}
REPAIRS = ('off', 'physical', 'physical-clip')


def offsets(grids):
    return np.cumsum([0]+[r*c for r, c in grids])


def planes(values, grids):
    off = offsets(grids)
    return [np.asarray(values)[off[i]:off[i+1]].reshape(shape)
            for i, shape in enumerate(grids)]


def resize_dct(plane, shape):
    coeffs = dctn(plane, norm='ortho')
    output = np.zeros(shape)
    r, c = min(shape[0], plane.shape[0]), min(shape[1], plane.shape[1])
    output[:r, :c] = coeffs[:r, :c]*np.sqrt(np.prod(shape)/plane.size)
    return idctn(output, norm='ortho')


def component_rgb(values, grids, transform, factor=1):
    shape = tuple(max(s[i] for s in grids)*factor for i in range(2))
    components = np.stack([(resize_dct(p, shape)+1)/2
                           for p in planes(values, grids)], -1)
    return transform.inverse(components)


def repair_components(values, grids, transform, source, masks, clip=False, reference_mean=None):
    shape = tuple(max(s[i] for s in grids) for i in range(2))
    source_light = linearize(source)@np.array([.2126, .7152, .0722])
    target_light = np.asarray(Image.fromarray(np.float32(source_light), 'F').resize(
        shape[::-1], Image.Resampling.BOX), float)
    working = [(p+1)/2 for p in planes(values, grids)]
    priors = ([np.zeros(s) for s in grids] if reference_mean is None else
              planes(reference_mean, grids))
    for _ in range(3):
        # Evaluate the picture the carried supports can actually produce,
        # including the agreed model means at omitted ordinary positions.
        shown = [idctn(np.where(mask, dctn(p*2-1, norm='ortho'), prior),
                       norm='ortho')/2+.5 for p, mask, prior in zip(working, masks, priors)]
        components = np.stack([resize_dct(p, shape) for p in shown], -1)
        rgb = transform.inverse(components)
        light = linearize(np.clip(rgb, 0, 1))@np.array([.2126, .7152, .0722])
        jac_rgb, jac_light = [], []
        for i in range(3):
            perturbed = components.copy()
            perturbed[..., i] += .001
            changed = transform.inverse(perturbed)
            jac_rgb.append((changed-rgb)/.001)
            jac_light.append((linearize(np.clip(changed, 0, 1))@
                              np.array([.2126, .7152, .0722])-light)/.001)
        gradient = np.stack(jac_light, -1)
        delta = ((target_light-light)/np.maximum((gradient*gradient).sum(-1), .01))[..., None]*gradient
        if clip:
            # Penalize RGB excursions only: no forced desaturation of in-gamut colors.
            jac = np.stack(jac_rgb, -1)
            error = np.clip(rgb, 0, 1)-rgb
            step = np.einsum('...rc,...r->...c', jac, error)
            step /= np.maximum(np.sum(jac*jac, axis=(-2, -1)), .01)[..., None]
            delta += step
        for i in range(3):
            correction = resize_dct(np.clip(delta[..., i], -.05, .05), grids[i])
            correction = idctn(dctn(correction, norm='ortho')*masks[i], norm='ortho')
            correction -= correction.mean()
            working[i] += correction
    return np.concatenate([p.ravel()*2-1 for p in working])


def analyze(rgb, transform, grids, shapes, selection=None, masks=None,
            repair='off', kernel_scope='all', reference_mean=None):
    x = np.asarray(rgb, float)/255
    converted = transform.forward(x)
    h, w = x.shape[:2]
    options = (selection.kernel.prefilter(selection.params) or {}
               if selection and selection.kernel.has_prefilter else {})
    full = options.get('full_source', False)
    factor = options.get('preshrink', 4)
    reference, projected = [], []
    for i, (rows, cols) in enumerate(grids):
        by = 1 if full else max(1, int(h/(factor*rows)+.5))
        bx = 1 if full else max(1, int(w/(factor*cols)+.5))
        nh, nw = -(-h//by), -(-w//bx)
        # Reuse the production block and anti-alias analysis geometry, but
        # with each component's own sampling dimensions.
        stack = np.repeat(converted[..., i:i+1], 3, axis=-1)
        mean = _block_means_f64(np.ascontiguousarray(stack), by, bx, nh, nw)[0]
        dy = not full and h % (2*by) == 0 and nh//2 >= 1.75*rows
        dx = not full and w % (2*bx) == 0 and nw//2 >= 1.75*cols
        if dy or dx:
            mean = _decimate(mean, dy, dx)
        reference.append(mean)
        left, right = _direct_plan(h, w, rows, cols, by, bx, int(dy), int(dx))[-2:]
        projected.append(_separable(left, np.ascontiguousarray(mean), right))
    frame = None
    if selection:
        frame = KernelFrame(selection, grids, shapes, masks, reference)
        if kernel_scope == 'all':
            # Kernel is a spatial operator on each component; this does not
            # declare any component to be physical brightness.
            for context in frame.contexts:
                context.plane = 0
        for i, grid in enumerate(projected):
            gain = frame.gain(i)
            if gain is not None:
                projected[i] = idctn(dctn(grid, norm='ortho')*gain, norm='ortho')
    values = np.concatenate([p.ravel()*2-1 for p in projected])
    if repair != 'off' and masks is not None and transform.base not in ('gray', 'hsv', 'hsl'):
        values = repair_components(values, grids, transform, x, masks,
                                   clip=repair == 'physical-clip', reference_mean=reference_mean)
    if frame:
        values = frame.post(values, grids)
        error = selection.kernel.take_error()
        if error:
            raise RuntimeError('kernel failure: '+error)
    return np.clip(values, -1, 1)


def fit(frames, transform, grids):
    off = offsets(grids)
    total, square, count = np.zeros(off[-1]), np.zeros(off[-1]), 0
    for rgb in frames:
        values = analyze(rgb, transform, grids, grids)
        c = np.concatenate([dctn(p, norm='ortho').ravel() for p in planes(values, grids)])
        total += c
        square += c*c
        count += 1
    if count < 2:
        raise ValueError('at least two training images are needed')
    mean = total/count
    floor = np.concatenate([1e-5/(1+(np.mgrid[:r, :c]**2).sum(0)).ravel()
                            for r, c in grids])
    return mean, np.maximum((square-total*total/count)/(count-1), floor)


def support(layout, grids, split):
    width, height = layout_size(layout)
    off = offsets(grids)
    kept = []
    for i, (shape, count) in enumerate(zip(grids, SPLITS[split])):
        if np.prod(shape) < count:
            raise ValueError('grid cannot hold its candidate coefficient pool')
        u, v = np.mgrid[:shape[0], :shape[1]]
        order = np.lexsort((v.ravel(), u.ravel(), (u*u*width*width+v*v*height*height).ravel()))
        kept.append(off[i]+order[:count])
    return np.concatenate(kept)


class ComponentCoder(AspectCoder):
    def __init__(self, positions, grids, shapes):
        self.positions = np.asarray(positions, np.int64)
        self.grids, self.shapes = list(grids), list(shapes)
        self.count, self.source_count = len(positions), int(offsets(grids)[-1])
        self.gains, self.variance = np.ones(self.count), None
        self.truncated = True
        self._grid = Grids(grids)


class ComponentFold(FoldCodec):
    """Same analog fold, with hosts/guests selected across actual components."""
    def __init__(self, model, grids, kept, hosts, guests, guest_variance, mono):
        self.model, self.M, self.filter = model, 500, 'box'
        self.conf_min, self.design_db, self.fitted_on = .9, 40., 'training-only component statistics'
        self.signature, self.noise_max = 16, .3
        self.grid, self.kept = Grids(grids), np.asarray(kept)
        self.hosts, self.guests = np.asarray(hosts), np.asarray(guests)
        self.sd_host, self.sd_guest = np.sqrt(model.lam[self.hosts]), np.sqrt(guest_variance)
        self.train = []
        self.sent_model_indices = np.unique(model.rank_tables[0][model.rank_tables[0] >= 0])
        self.set_step(1.)
        self.use_compand(12., 4., 1., .15 if mono else .1)
        self._set_identity()


class ComponentAspect(AspectFoldWire):
    def __init__(self, model, codec, layout):
        super().__init__(layout, 'fixed')
        self.model, self._codecs = model, {id(model): codec}

    def model_for(self, base_model, layout):
        if layout != self.layout:
            raise ValueError('agreed component layout mismatch')
        return self.model


class ComponentMono(AspectMonoWire):
    def __init__(self, base, model, codec, layout):
        self.layout, self.model, self.fold_codec = layout, model, codec
        self._aspect = ComponentAspect(model, codec, layout)
        MonoFreshFoldWire.__init__(self, base, side='left')

    def model_for(self, model, layout=None):
        if layout is not None and layout != self.layout:
            raise ValueError('agreed component layout mismatch')
        self._codec_by_model_id[id(self.model)] = self.fold_codec
        return self.model


class AnalogFrameWire(ColorWire):
    def __init__(self, profile, layout, transform, statistics, grid='equal',
                 split='current', kernel='matched_viewer', repair='off', kernel_scope='all'):
        self.profile, self.layout, self.transform = profile, layout, transform
        self.grids, self.grid_name = GRID_FAMILIES[grid], grid
        self.repair, self.kernel_scope, self.split = repair, kernel_scope, split
        self.production, self.kernel_name = False, kernel
        self.base = v7.load_model(TARGET, 'box')
        self.selection = v7_kernels.open_registry([
            Path(__file__).with_name('v7_analog_kernels')]).select(kernel, profile=profile)
        off = offsets(self.grids)
        mean, variance = statistics
        kept = support(layout, self.grids, split)
        self.reference_mean = np.zeros(off[-1])
        self.reference_mean[kept] = mean[kept]
        plane = np.searchsorted(off[1:], kept, side='right')
        lam = variance[kept]
        dc = np.flatnonzero(np.isin(kept, off[:3]))
        rest = np.argsort(-lam, kind='stable')
        order = np.concatenate((dc, rest[~np.isin(rest, dc)]))
        gain = lam**-.25
        end = 1264 if profile == 'aspect-mono-500' else v7.BODY_END
        gain /= np.sqrt(np.mean((gain*gain*lam)[order[:end]]))
        tables = {'positions': kept, 'mu': mean[kept], 'lam': lam, 'gain': gain,
                  'order': order, 'tail_luma_slots': np.int64(96), 'unit_rms': np.float64(1)}
        model = _assemble(tables, self.base.phase, 1., template=self.base)
        shapes = []
        for i, shape in enumerate(self.grids):
            local = kept[plane == i]-off[i]
            shapes.append((int((local//shape[1]).max())+1, int((local % shape[1]).max())+1))
        coder = ComponentCoder(kept, self.grids, shapes)
        model = replace(model, coder=coder, plane=plane)
        mono = profile == 'aspect-mono-500'
        if mono:
            ranks = fresh_rank_tables(model, order)
            priors = tuple(v7.block_priors(gain, lam, rank) for rank in ranks)
            model = replace(model, rank_tables=ranks, block_prior_tables=priors,
                            block_prior_tables32=tuple(np.float32(x) for x in priors))
            model._mono_wire_profile = profile
        synth = np.random.default_rng(v7.LEVEL_SEED).standard_normal(2880)*np.sqrt(lam)+model.mu
        audio = v7.encode_frame_coeffs(model, synth, 1)
        model = replace(model, scale=float(TARGET/np.sqrt(np.mean(audio*audio)))*
                        (np.sqrt(2) if mono else 1))
        sent = order[:1264 if mono else v7.BODY_END+v7.TAIL_PER]
        body = order[v7.HEAD:1264 if mono else v7.BODY_END]
        hosts = body[-500:]
        if mono:
            guest_indices = order[1264:1764]
            guests, guest_lam = kept[guest_indices], lam[guest_indices]
        else:
            available = np.flatnonzero(~np.isin(np.arange(off[-1]), kept))
            guests = available[np.argsort(-variance[available], kind='stable')[:500]]
            guest_lam = variance[guests]
        codec = ComponentFold(model, self.grids, kept, hosts, guests, guest_lam, mono)
        codec.layout = layout
        self.model, self.codec = model, codec
        self.wire = (ComponentMono(self.base, model, codec, layout) if mono else
                     ComponentAspect(model, codec, layout))
        mask = np.zeros(off[-1], bool)
        mask[kept[sent]] = True
        mask[kept[hosts[-16:]]] = False
        mask[guests[:-16]] = True
        self.masks = planes(mask, self.grids)
        self.shapes = shapes
        self.aspect = v7.V7_ASPECT_NAMES.index(layout)
        self.tables = tables

    def values(self, rgb):
        return analyze(rgb, self.transform, self.grids, self.shapes,
                       self.selection, self.masks, self.repair, self.kernel_scope,
                       self.reference_mean)

    def reconstruct(self, values, factor=1):
        return component_rgb(values, self.grids, self.transform, factor)

    def record(self):
        record = super().record()
        record.update(grids=self.grids, sent_bounds=self.shapes, grid_family=self.grid_name,
                      repair=self.repair, kernel_scope=self.kernel_scope,
                      component_hosts='all components eligible', temporal_picture_memory=False,
                      carried_positions=np.flatnonzero(np.concatenate(
                          [p.ravel() for p in self.masks])).tolist(),
                      signature_hosts=self.codec.kept[self.codec.hosts[-16:]].tolist(),
                      brightness_repair=self.repair)
        return record


class ProductionFrameWire(ColorWire):
    """Production control, including its existing luma-repair options."""
    def __init__(self, profile, layout, repair='ordinary'):
        from tools.v7_send_gui import default_kernel_for_profile
        super().__init__(profile, layout, ColorTransform('pillow-ycbcr'),
                         kernel=default_kernel_for_profile(profile))
        self.repair = repair

    def values(self, rgb):
        options = {'kernel': self.selection, 'linear_light': self.repair == 'linear'}
        values = v7_live._values(
            self.base, rgb, 'box', brightness=1., gamma=1., dct_encode=True,
            dct_options=options,
            chroma_sent_for=(None if self.repair == 'off' else lambda _: self.masks[1:]),
            kernel_masks_for=lambda _: self.masks)[0]
        if self.repair == 'clip':
            from animation_modem.v7_source_dct import clip_aware_luma
            values = clip_aware_luma(values, v7.V7_GRIDS[0], self.codec.sent_luma_mask())
        return values

    def record(self):
        record = super().record()
        record['brightness_repair'] = self.repair
        return record


def render(wire, values, size, display=False):
    if hasattr(wire, 'render_received'):
        return wire.render_received(values, size, display)
    from tools.v7_color_metrics import resize_rgb, mitchell_resize, display_defaults
    from tools.v7_gl_viewer import dct_reconstruct_planes
    if isinstance(wire, AnalogFrameWire):
        rgb = wire.reconstruct(values)
        if not display:
            return resize_rgb(rgb, size)
        # No reduction to the old color grid. All physical-display planes
        # retain the entire decoded RGB lattice, then enlarge consistently.
        transform = ColorTransform('ycbcr601')
        working = transform.forward(rgb)*2-1
        p = [working[..., i] for i in range(3)]
        defaults = display_defaults()
        p = dct_reconstruct_planes(p, defaults['dct'], edge=defaults['edge'],
                                   edge_strength=defaults['edge_strength'], chroma='off')
        return transform.inverse(np.stack([(mitchell_resize(x, size)+1)/2 for x in p], -1))
    from tools.v7_color_metrics import render as production_render
    return production_render(wire, values, size, display)
