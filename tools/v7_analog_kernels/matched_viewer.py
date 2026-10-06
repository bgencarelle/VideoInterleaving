"""Grid-aware compensation for 4x DCT reconstruction and Mitchell sampling."""
import numpy as np

LABEL = 'Matched analog viewer'
HELP = 'Compensate the viewer at the actual decoded component grid, without a sent-size box reduction.'
PARAMS = {'strength': (1., 0., 1., .1), 'up': (4, 2, 8, 1, 'DCT reconstruction factor', True)}


def _response(nu):
    x = np.linspace(-2., 2., 257)
    a = abs(x)
    w = np.where(a < 1, (7*a**3-12*a**2+16/3)/6,
                 np.where(a < 2, (-7/3*a**3+12*a**2-20*a+32/3)/6, 0))
    return np.sum(w[:, None]*np.cos(2*np.pi*x[:, None]*nu.ravel()[None]), axis=0).reshape(nu.shape)/w.sum()


def gain(ctx, strength, up):
    rows, cols = ctx.grid
    fy = np.arange(rows)[:, None]/(2*rows*int(up))
    fx = np.arange(cols)[None, :]/(2*cols*int(up))
    response = _response(fy)*_response(fx)
    return 1+strength*(np.clip(1/np.maximum(response, .5), 1., 1.15)-1)
