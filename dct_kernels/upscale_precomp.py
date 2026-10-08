"""Undo the blur the receiver's upscaler adds, before the signal is sent."""
import numpy as np                 # the gain window only, built once per setting

LABEL = 'Upscaler pre-compensation'
HELP = ('The picture is small and the viewer blows it up; every upscaler '
        'softens it (bilinear most, then bicubic). This kernel boosts the '
        'band by the inverse of that softness (a Wiener inverse, so noise is '
        'not boosted without limit) and then moves the halos it causes back '
        'inside the source range. Pick the upscaler your viewer uses: '
        '0 bilinear, 1 bicubic (Catmull-Rom), 2 nearest, 3 none (ideal).')
PARAMS = {
    'upscaler': (0, 0, 3, 1, '0 bilinear, 1 bicubic, 2 nearest, 3 ideal', True),
    'strength': (1.0, 0.0, 2.0, 0.05, 'how much of the inverse to apply'),
    'noise': (0.15, 0.005, 1.0, 0.01, 'Wiener term: larger = gentler boost'),
    'max_gain': (2.2, 1.0, 6.0, 0.1, 'ceiling on the boost at any frequency'),
    'tame': (6, 0, 30, 1, 'halo clean-up rounds after the boost (0 = off)', True),
}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {'luma_mix': 0.25, 'tame': 0},
    'aspect-fold-500': {'luma_mix': 0.2, 'tame': 0},
    # The stereo nested fold's default kernel, as it was benchmarked: the
    # full inverse (these were the kernel's raw defaults, applied because no
    # entry existed).
    'stereo-nested': {'upscaler': 0, 'strength': 1.0, 'noise': 0.15,
                      'max_gain': 2.2, 'tame': 6, 'luma_mix': 1.0,
                      'chroma_mix': 1.0},
}

_X = np.linspace(-2.0, 2.0, 1025)


def _catmull_rom(x):
    x = np.abs(x)
    return np.where(x < 1, 1.5*x**3 - 2.5*x**2 + 1,
                    np.where(x < 2, -0.5*x**3 + 2.5*x**2 - 4*x + 2, 0.0))


def _droop(nu, upscaler):
    """Gain of the upscaler's interpolation kernel at ``nu`` cycles/pixel."""
    if upscaler == 0:
        return np.sinc(nu)**2
    if upscaler == 2:
        return np.sinc(nu)
    if upscaler == 1:
        k = _catmull_rom(_X)
        step = _X[1] - _X[0]
        flat = np.ravel(nu)                       # nu may be (rows, 1) or (1, cols)
        table = np.abs(np.cos(2*np.pi*np.outer(flat, _X)) @ k)*step
        return table.reshape(np.shape(nu))
    return np.ones_like(nu)


def response(nu, upscaler, strength, noise, max_gain):
    nu = np.asarray(nu, float)
    h = _droop(nu, int(upscaler))
    inverse = h/(h*h + noise)
    inverse = inverse/(1.0/(1.0 + noise))          # 1 at DC
    gain = 1.0 + strength*(inverse - 1.0)
    return np.clip(gain, 0.0, max_gain)


def post(grid, ctx, tame):
    if ctx.reference is None or int(tame) < 1 or ctx.plane:
        return grid
    return ctx.tame(grid, tame)
