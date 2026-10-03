"""Pluggable DCT-mode downscale kernels, discovered from folders at launch.

A kernel is one ``.py`` file. Drop it in ``dct_kernels/`` (or any folder named
by ``--dct-kernel-dir`` or ``$V7_KERNEL_DIR``) and it is offered the next time
the sender or its GUI starts; delete the file and it is gone. Nothing here
changes the wire: a kernel only decides which picture is handed to the same
DCT, fold and modem. The picture is prepared as before (block mean, 2:1
decimation, DCT truncation to the coder grid); a kernel shapes the result in
three places, all optional:

``gain`` / ``response`` / ``kernel``  (a linear window over the DCT)
    The sent band is the part of the spectrum the wire carries. A window is a
    gain per DCT coefficient. Write whichever of these is easiest:

    * ``kernel(x, **params)``: a 1-D resampling kernel in *sent-pixel* units
      (``x = 1`` is one pixel of the picture the wire can hold). Set
      ``SUPPORT`` to its half-width. The host works out its frequency
      response, so Lanczos, Mitchell, B-splines and so on are three lines.
    * ``response(nu, **params)``: a 1-D frequency response; ``nu`` is in
      cycles per sent pixel (0.5 is the Nyquist of what the wire holds).
      Applied along both axes; set ``RADIAL = True`` to apply it to the
      radius instead.
    * ``gain(ctx, **params)``: a full 2-D array over the coder grid.

    The gain at DC is forced to 1, so brightness never moves. With luma
    adjustment on, the same window is applied to the luminance it aims at, so
    the two agree.

``post(grid, ctx, **params)``  (a non-linear refit)
    Runs last, on the final values of one plane in [0, 1], after luma
    adjustment. ``ctx.reference`` is the plane before the window, ``ctx.mask``
    the coefficients the wire carries, ``ctx.project(grid)`` keeps only those
    and ``ctx.reduce(array, 'min' | 'max' | 'mean')`` brings a pixel-domain
    array down to the grid. Return a grid of the same shape.

Optional module attributes: ``LABEL``, ``HELP``, ``NAME`` (default: the file
name), ``SUPPORT``, ``RADIAL`` and ``PARAMS``, a dict of name to ``Param`` (or
a ``(default, low, high, step, help, integer)`` tuple, the last two optional). Every kernel also gets
``luma_mix`` and ``chroma_mix``: 0 switches the kernel off for that plane, 1
is as written, above 1 pushes further.

A kernel that fails to import, or to run on a test picture, is listed in
``registry.errors`` and left out; one that fails mid-stream is bypassed so the
picture keeps going.
"""
import hashlib
import inspect
import os
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn

REFERENCE = 'reference'
DEFAULT_DIR = Path(__file__).resolve().parents[1]/'dct_kernels'
ENV_DIRS = 'V7_KERNEL_DIR'
HOST_PARAMS = ('luma_mix', 'chroma_mix')
RESERVED = frozenset(HOST_PARAMS) | {'ctx', 'plane', 'grid', 'nu', 'x'}
GAIN_LIMITS = (-2.0, 8.0)
PRESHRINK_LIMITS = (1.75, 8.0)
SLOW_POST_MS = 25.0           # a refit this slow eats a third of a 12 fps frame
SLOW_BUILD_MS = 100.0         # a window this slow to build stalls a slider drag
CACHE_LIMIT = 256
_INTEGRAL_POINTS = 2049


class KernelError(Exception):
    """A kernel file that cannot be used, or a kernel that failed to run."""


@dataclass(frozen=True)
class Param:
    """One live-adjustable number of a kernel."""
    default: float
    low: float = 0.0
    high: float = 1.0
    step: float = 0.05
    help: str = ''
    integer: bool = False

    @classmethod
    def coerce(cls, name, spec):
        if isinstance(spec, cls):
            param = spec
        elif isinstance(spec, (tuple, list)) and 3 <= len(spec) <= 6:
            param = cls(*spec)
        elif isinstance(spec, (int, float)) and not isinstance(spec, bool):
            param = cls(float(spec), 0.0, max(1.0, 2*abs(float(spec))))
        else:
            raise KernelError(f'parameter {name!r} is not a Param')
        values = (param.default, param.low, param.high, param.step)
        if not all(isinstance(v, (int, float)) and np.isfinite(v)
                   for v in values):
            raise KernelError(f'parameter {name!r} has a non-numeric range')
        if not param.low <= param.default <= param.high or param.step <= 0:
            raise KernelError(f'parameter {name!r} default is outside its range')
        return param

    def clamp(self, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError('not finite')
        value = min(max(value, self.low), self.high)
        return float(round(value)) if self.integer else value


HOST_PARAM_SPECS = {
    'luma_mix': Param(1.0, 0.0, 2.0, 0.05,
                      '0 = this kernel off for luma, 1 = as written'),
    'chroma_mix': Param(1.0, 0.0, 2.0, 0.05,
                        '0 = this kernel off for chroma, 1 = as written'),
}


class KernelContext:
    """What a kernel is told about the plane it is working on."""

    def __init__(self, plane, grid, sent, mask=None, reference=None):
        self.plane = int(plane)               # 0 luma, 1 Cb, 2 Cr
        self.grid = (int(grid[0]), int(grid[1]))     # coder grid (rows, cols)
        self.sent = (int(sent[0]), int(sent[1]))     # transmitted rectangle
        if mask is None:
            mask = np.zeros(self.grid, bool)
            mask[:self.sent[0], :self.sent[1]] = True
        mask = np.asarray(mask, bool).reshape(self.grid)
        mask.setflags(write=False)
        self.mask = mask
        rows_on, cols_on = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
        # Smallest rectangle holding every coefficient the wire carries: the
        # sent rectangle plus the folded guests.
        self.extent = ((int(rows_on[-1])+1, int(cols_on[-1])+1)
                       if len(rows_on) and len(cols_on) else self.sent)
        self.reference = reference            # pre-window plane, post only
        self.mask_key = hashlib.blake2b(mask.tobytes(), digest_size=8).digest()

    def nu(self):
        """(rows x 1, 1 x cols) frequencies in cycles per sent pixel."""
        u = np.arange(self.grid[0], dtype=np.float64)[:, None]
        v = np.arange(self.grid[1], dtype=np.float64)[None, :]
        return u/(2.0*self.sent[0]), v/(2.0*self.sent[1])

    def band(self, extent=False):
        """Radius in sent-band units: 1.0 at the edge of the sent rectangle,
        or with ``extent`` of the rectangle that also holds the guests."""
        rows, cols = self.extent if extent else self.sent
        u = np.arange(self.grid[0], dtype=np.float64)[:, None]
        v = np.arange(self.grid[1], dtype=np.float64)[None, :]
        return np.hypot(u/rows, v/cols)

    def project(self, grid):
        """The picture the receiver can show: only the coefficients sent."""
        coefficients = dctn(np.asarray(grid, np.float64), norm='ortho')
        coefficients[~self.mask] = 0.0
        return idctn(coefficients, norm='ortho')

    def reduce(self, array, how='mean'):
        """A pixel-domain array brought down to the grid (mean, min or max
        over the source pixels each grid pixel covers)."""
        array = np.asarray(array, np.float64)
        for axis, count in enumerate(self.grid):
            array = _reduce_axis(array, axis, count, how)
        return array


def _reduce_axis(array, axis, count, how):
    """Reduce one axis to ``count`` cells; each covers the source pixels it
    touches (neighbouring cells may share a boundary pixel)."""
    ufunc = {'min': np.minimum, 'max': np.maximum, 'mean': np.add}.get(how)
    if ufunc is None:
        raise ValueError("reduce(how) is 'mean', 'min' or 'max'")
    size = array.shape[axis]
    if size == count:
        return array
    edges = np.linspace(0, size, count+1)
    first = np.minimum(np.floor(edges[:-1] + 1e-9).astype(np.intp), size-1)
    last = np.minimum(np.ceil(edges[1:] - 1e-9).astype(np.intp), size)
    last = np.maximum(last, first+1)
    # reduceat over [first0, last0, first1, last1, ...]: the even entries are
    # the cells, the odd ones (gaps or overlaps) are dropped. The final cell
    # runs to the end of the axis, which is where `last` ends.
    indices = np.empty(2*count-1, dtype=np.intp)
    indices[0::2] = first
    indices[1::2] = last[:-1]
    reduced = np.take(ufunc.reduceat(array, indices, axis=axis),
                      range(0, 2*count-1, 2), axis=axis)
    if how == 'mean':
        shape = [1]*array.ndim
        shape[axis] = count
        reduced = reduced/(last-first).reshape(shape)
    return reduced


def _accepted(function, skip=1):
    """Parameter names a hook takes, or None when it takes any (**kwargs)."""
    names = []
    for parameter in list(inspect.signature(function).parameters.values())[skip:]:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY):
            names.append(parameter.name)
    return frozenset(names)


class Kernel:
    """One loaded kernel file."""

    def __init__(self, name, module, path):
        self.name = name
        self.module = module
        self.path = Path(path) if path is not None else None
        self.stamp = _stamp(self.path)
        self.label = str(getattr(module, 'LABEL', name))
        self.help = str(getattr(module, 'HELP', '')).strip()
        self.radial = bool(getattr(module, 'RADIAL', False))
        self.support = float(getattr(module, 'SUPPORT', 3.0))
        declared = getattr(module, 'PARAMS', {}) or {}
        if not isinstance(declared, dict):
            raise KernelError('PARAMS must be a dict')
        if len(declared) > 12:
            raise KernelError('a kernel may have at most 12 parameters')
        self.params = {}
        for key, spec in declared.items():
            if not isinstance(key, str) or not key.isidentifier() \
                    or key in RESERVED:
                raise KernelError(f'parameter name {key!r} is not allowed')
            self.params[key] = Param.coerce(key, spec)
        self.params.update(HOST_PARAM_SPECS)
        self._hooks = {}
        for hook in ('gain', 'response', 'kernel', 'post', 'prefilter'):
            function = getattr(module, hook, None)
            if function is None:
                continue
            if not callable(function):
                raise KernelError(f'{hook} is not callable')
            self._hooks[hook] = (function, _accepted(
                function, 0 if hook == 'prefilter' else 1))
        if not self._hooks:
            raise KernelError('defines none of gain, response, kernel, post '
                              'or prefilter')
        if not 0.25 <= self.support <= 16:
            raise KernelError('SUPPORT must be between 0.25 and 16')
        self._cache = {}
        self._lock = threading.Lock()
        self._error = None
        self.failures = 0
        self.post_ms = 0.0            # smoothed time of the refit per call
        self.build_ms = 0.0           # time of the last window built
        self._warned_slow = False

    @property
    def has_gain(self):
        return any(hook in self._hooks for hook in ('gain', 'response', 'kernel'))

    @property
    def has_post(self):
        return 'post' in self._hooks

    @property
    def has_prefilter(self):
        return 'prefilter' in self._hooks

    def prefilter(self, params):
        """Options for the stage before the DCT: ``{'preshrink': factor}``.

        ``factor`` is how many times the luma grid the block-averaged plane
        is (1.75 to 8; the shipped encoder uses 4). Anything else a kernel
        returns is ignored. None when the kernel does not choose.
        """
        if 'prefilter' not in self._hooks:
            return None
        function, accepted = self._hooks['prefilter']
        arguments = {name: value for name, value in params.items()
                     if name not in HOST_PARAMS and
                     (accepted is None or name in accepted)}
        result = function(**arguments)
        if result is None:
            return None
        if not isinstance(result, dict):
            raise KernelError('prefilter must return a dict or None')
        options = {}
        if 'preshrink' in result:
            value = float(result['preshrink'])
            if not np.isfinite(value) or not PRESHRINK_LIMITS[0] <= value <= PRESHRINK_LIMITS[1]:
                raise KernelError('preshrink must be between '
                                  f'{PRESHRINK_LIMITS[0]:g} and {PRESHRINK_LIMITS[1]:g}')
            options['preshrink'] = value
        return options

    def defaults(self):
        return {name: spec.default for name, spec in self.params.items()}

    def resolve(self, values=None):
        """Every parameter, clamped to its range; unknown names are refused."""
        resolved = self.defaults()
        for name, value in (values or {}).items():
            if name not in self.params:
                raise KernelError(f'{self.name} has no parameter {name!r}')
            try:
                resolved[name] = self.params[name].clamp(value)
            except (TypeError, ValueError):
                raise KernelError(f'{self.name}.{name} must be a finite number')
        return resolved

    # ---------------------------------------------------------------- calls
    def _call(self, hook, first, params, plane):
        function, accepted = self._hooks[hook]
        arguments = {name: value for name, value in params.items()
                     if name not in HOST_PARAMS and
                     (accepted is None or name in accepted)}
        if accepted is None or 'plane' in accepted:
            arguments['plane'] = plane
        return function(first, **arguments)

    def _spatial_response(self, nu, params, plane):
        x = np.linspace(-self.support, self.support, _INTEGRAL_POINTS)
        taps = np.asarray(self._call('kernel', x, params, plane), np.float64)
        if taps.shape != x.shape or not np.isfinite(taps).all():
            raise KernelError('kernel(x) must return a finite array like x')
        total = taps.sum()
        if abs(total) < 1e-9:
            raise KernelError('kernel has no DC gain')
        flat = np.asarray(nu, np.float64).reshape(-1)
        response = np.cos(2*np.pi*np.outer(flat, x)) @ taps / total
        return response.reshape(np.shape(nu))

    def _raw_gain(self, ctx, params):
        if 'gain' in self._hooks:
            return self._call('gain', ctx, params, ctx.plane)
        nu_rows, nu_cols = ctx.nu()
        if 'response' in self._hooks:
            def axis(nu):
                return np.asarray(self._call('response', nu, params, ctx.plane),
                                  np.float64)
        else:
            def axis(nu):
                return self._spatial_response(nu, params, ctx.plane)
        if self.radial:
            return axis(np.hypot(nu_rows, nu_cols))
        return axis(nu_rows) * axis(nu_cols)

    def gain(self, ctx, params):
        """The 2-D gain over ``ctx.grid`` for this plane, or None (no window).

        Cached on everything that can change it, so a live slider only
        rebuilds the one array it touched.
        """
        if not self.has_gain:
            return None
        mix = params['luma_mix' if ctx.plane == 0 else 'chroma_mix']
        if mix == 0.0:
            return None
        key = (ctx.plane, ctx.grid, ctx.sent, ctx.mask_key,
               tuple(sorted(params.items())))
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        started = time.perf_counter()
        gain = np.broadcast_to(np.asarray(self._raw_gain(ctx, params),
                                          np.float64), ctx.grid).copy()
        self.build_ms = (time.perf_counter()-started)*1000
        if not np.isfinite(gain).all():
            raise KernelError('gain is not finite')
        gain[0, 0] = 1.0
        gain = 1.0 + mix*(gain - 1.0)
        gain = np.clip(gain, *GAIN_LIMITS)
        gain.setflags(write=False)
        with self._lock:
            if len(self._cache) >= CACHE_LIMIT:
                self._cache.clear()
            self._cache[key] = gain
        return gain

    def post(self, grid, ctx, params):
        """The refit grid for this plane, or the input when there is none."""
        if 'post' not in self._hooks:
            return grid
        mix = params['luma_mix' if ctx.plane == 0 else 'chroma_mix']
        if mix == 0.0:
            return grid
        started = time.perf_counter()
        result = np.asarray(self._post_call(grid, ctx, params), np.float64)
        elapsed = (time.perf_counter()-started)*1000
        self.post_ms = elapsed if not self.post_ms else .8*self.post_ms+.2*elapsed
        if result.shape != grid.shape or not np.isfinite(result).all():
            raise KernelError('post must return a finite grid of the same shape')
        if np.abs(result).max() > 8:
            raise KernelError('post returned values far outside [0, 1]')
        return grid + mix*(result - grid) if mix != 1.0 else result

    def _post_call(self, grid, ctx, params):
        function, accepted = self._hooks['post']
        arguments = {name: value for name, value in params.items()
                     if name not in HOST_PARAMS and
                     (accepted is None or name in accepted)}
        return function(grid, ctx, **arguments)

    # --------------------------------------------------------------- errors
    def note_failure(self, exc):
        self.failures += 1
        self._error = f'{self.name}: {type(exc).__name__}: {exc}'

    def take_error(self):
        error, self._error = self._error, None
        return error

    def take_slow_warning(self):
        """Once, a message if this kernel is heavy enough to hurt a live send."""
        if self._warned_slow:
            return None
        if self.post_ms > SLOW_POST_MS:
            what = f'its refit takes {self.post_ms:.0f} ms per frame'
        elif self.build_ms > SLOW_BUILD_MS:
            what = f'it takes {self.build_ms:.0f} ms to rebuild after a change'
        else:
            return None
        self._warned_slow = True
        return f'DCT kernel {self.name} is slow: {what}'


@dataclass(frozen=True)
class KernelSelection:
    """A kernel with the parameter values to run it at (immutable, hashable)."""
    kernel: Kernel
    values: tuple

    @property
    def params(self):
        return dict(self.values)


def _stamp(path):
    if path is None:
        return None
    try:
        # Content, not just mtime: two saves inside one timestamp tick (or a
        # same-length edit) must still count as a change.
        return hashlib.blake2b(path.read_bytes(), digest_size=8).digest()
    except OSError:
        return None


def _selftest(kernel):
    """Run every hook once on small pictures so a broken file is refused at
    launch instead of stalling a live send."""
    params = kernel.defaults()
    rng = np.random.default_rng(7)
    for plane, grid, sent in ((0, (96, 80), (48, 40)),
                              (1, (48, 40), (24, 20))):
        mask = np.zeros(grid, bool)
        mask[:sent[0], :sent[1]] = True
        reference = rng.random((grid[0]*2, grid[1]*2))
        ctx = KernelContext(plane, grid, sent, mask, reference)
        gain = kernel.gain(ctx, params)
        if gain is not None and gain.shape != grid:
            raise KernelError('gain has the wrong shape')
        values = rng.random(grid)
        kernel.post(values, ctx, params)
    kernel.prefilter(params)


def load_kernel(path):
    """Import one kernel file; raises KernelError if it cannot be used."""
    path = Path(path)
    stem = path.stem
    module_name = f'v7_kernel_{stem}_{abs(hash((str(path), _stamp(path)))):x}'
    # Compiled here rather than imported: Python's bytecode cache is keyed on
    # the file's mtime in whole seconds and its size, so a same-length edit
    # saved within a second would silently run the old code, and the folder
    # would fill with __pycache__.
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    try:
        source = path.read_text(encoding='utf-8')
        exec(compile(source, str(path), 'exec'), module.__dict__)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise KernelError(f'import failed: {type(exc).__name__}: {exc}')
    name = str(getattr(module, 'NAME', stem))
    if not name.replace('-', '_').replace('.', '_').isidentifier() \
            or name == REFERENCE:
        sys.modules.pop(module_name, None)
        raise KernelError(f'name {name!r} is not allowed')
    try:
        kernel = Kernel(name, module, path)
        _selftest(kernel)
    except KernelError:
        sys.modules.pop(module_name, None)
        raise
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise KernelError(f'self-test failed: {type(exc).__name__}: {exc}')
    return kernel


class KernelRegistry:
    """The kernels found in a list of folders (later folders win)."""

    def __init__(self, dirs=()):
        self.dirs = []
        for folder in dirs:
            self.add_dir(folder)
        self._kernels = {}
        self.errors = []          # (file, message)
        self._lock = threading.Lock()

    def add_dir(self, folder):
        folder = Path(folder).expanduser()
        if folder not in self.dirs:
            self.dirs.append(folder)

    def scan(self):
        """(Re)read every folder. Files unchanged since the last scan keep
        their loaded kernel (and its caches). Returns (added, removed)."""
        previous = self._kernels
        found, errors = {}, []
        for folder in self.dirs:
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob('*.py')):
                if path.name.startswith(('_', '.')):
                    continue
                old = next((k for k in previous.values()
                            if k.path == path and k.stamp == _stamp(path)), None)
                if old is not None:
                    found[old.name] = old
                    continue
                try:
                    kernel = load_kernel(path)
                except KernelError as exc:
                    errors.append((str(path), str(exc)))
                    continue
                if kernel.name in found:
                    errors.append((str(path), f'replaces {found[kernel.name].path}'))
                found[kernel.name] = kernel
        with self._lock:
            self._kernels = found
            self.errors = errors
        return (sorted(set(found) - set(previous)),
                sorted(set(previous) - set(found)))

    def names(self):
        return [REFERENCE] + sorted(self._kernels)

    def get(self, name):
        if name in (None, '', REFERENCE):
            return None
        try:
            return self._kernels[name]
        except KeyError:
            raise KernelError(f'no kernel named {name!r} '
                              f'(have: {", ".join(self.names())})')

    def select(self, name, values=None):
        """A KernelSelection, or None for the reference (no kernel)."""
        kernel = self.get(name)
        if kernel is None:
            return None
        return KernelSelection(kernel, tuple(sorted(
            kernel.resolve(values).items())))

    def describe(self):
        """[(name, label, help, {param: Param})] for a UI."""
        rows = [(REFERENCE, 'Reference · the current encode',
                 'The shipped encode, unchanged.', {})]
        rows += [(k.name, k.label, k.help, dict(k.params))
                 for k in (self._kernels[n] for n in sorted(self._kernels))]
        return rows


def kernel_dirs(extra=()):
    """The default folder, then $V7_KERNEL_DIR entries, then ``extra``."""
    dirs = [DEFAULT_DIR]
    dirs += [Path(p) for p in os.environ.get(ENV_DIRS, '').split(os.pathsep) if p]
    dirs += [Path(p) for p in extra]
    return dirs


def open_registry(extra=()):
    """A scanned registry over the usual folders."""
    registry = KernelRegistry(kernel_dirs(extra))
    registry.scan()
    return registry
