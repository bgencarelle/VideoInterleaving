"""Explicit experimental color transforms for the wire-backed V7 benchmark.

All working planes have fixed scaling, independent of the current picture.
PCA is fitted on training pixels only; no whitening or per-picture fitting.
RGB inputs/outputs are float sRGB code values, ordinarily in [0, 1].
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image


LINEAR_NAMES = ('rgb', 'ycbcr601', 'ycbcr709', 'ycocg', 'opponent', 'pca')
NAMES = ('pillow-ycbcr',) + LINEAR_NAMES + tuple(
    name+'-linear' for name in LINEAR_NAMES) + (
        'ycocg-r', 'rct', 'xyz', 'lab', 'oklab', 'ictcp', 'hsv', 'hsl', 'gray')
XYZ = np.array([[.4124564, .3575761, .1804375],
                [.2126729, .7151522, .0721750],
                [.0193339, .1191920, .9503041]])
WHITE = XYZ.sum(axis=1)
OK_M1 = np.array([[.4122214708, .5363325363, .0514459929],
                  [.2119034982, .6806995451, .1073969566],
                  [.0883024619, .2817188376, .6299787005]])
OK_M2 = np.array([[.2104542553, .7936177850, -.0040720468],
                  [1.9779984951, -2.4285922050, .4505937099],
                  [.0259040371, .7827717662, -.8086757660]])
ICT_LMS = np.array([[1688, 2146, 262], [683, 2951, 462],
                    [99, 309, 3688]], float)/4096
ICT_M = np.array([[2048, 2048, 0], [6610, -13613, 7003],
                  [17933, -17390, -543]], float)/4096
RGB2020 = np.array([[.6369580483, .1446169036, .1688809752],
                   [.2627002120, .6779980715, .0593017165],
                   [0, .0280726930, 1.0609850577]])


def linearize(x):
    x = np.asarray(x, float)
    return np.where(x <= .04045, x/12.92,
                    np.maximum((x+.055)/1.055, 0)**2.4)


def delinearize(x):
    x = np.asarray(x, float)
    return np.where(x <= .0031308, 12.92*x,
                    1.055*np.maximum(x, 0)**(1/2.4)-.055)


def _pq(x, inverse=False):
    # Signed extension is explicit: it keeps noisy inverse samples finite.
    m1, m2 = 2610/16384, 2523/32
    c1, c2, c3 = 3424/4096, 2413/128, 2392/128
    s, a = np.sign(x), np.abs(x)
    if inverse:
        p = np.minimum(a, 1.2)**(1/m2)
        return s*(np.maximum(p-c1, 0)/np.maximum(c2-c3*p, 1e-9))**(1/m1)
    p = a**m1
    return s*((c1+c2*p)/(1+c3*p))**m2


def _matrix(name):
    if name == 'rgb':
        return np.eye(3)
    if name.startswith('ycbcr'):
        kr, kb = (.299, .114) if name == 'ycbcr601' else (.2126, .0722)
        kg = 1-kr-kb
        y = np.array([kr, kg, kb])
        return np.array([y, (np.array([0, 0, 1])-y)/(2*(1-kb)),
                         (np.array([1, 0, 0])-y)/(2*(1-kr))])
    if name == 'ycocg':
        return np.array([[.25, .5, .25], [.5, 0, -.5], [-.25, .5, -.25]])
    if name == 'opponent':
        return np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]]) / np.sqrt(
            np.array([3, 2, 6]))[:, None]
    raise ValueError(name)


@dataclass
class ColorTransform:
    name: str
    matrix: object = None
    mean: object = None

    def __post_init__(self):
        if self.name not in NAMES:
            raise ValueError(f'unknown transform {self.name}')
        self.linear = self.name.endswith('-linear')
        self.base = self.name.removesuffix('-linear')
        if self.base in LINEAR_NAMES and self.base != 'pca':
            self.matrix = _matrix(self.base)
            self.mean = np.zeros(3)
        if self.matrix is not None:
            self.matrix = np.asarray(self.matrix, float)
            self.mean = np.asarray(self.mean, float)
            self.low = np.minimum(self.matrix, 0).sum(axis=1)-self.matrix@self.mean
            self.high = np.maximum(self.matrix, 0).sum(axis=1)-self.matrix@self.mean
            self.inverse_matrix = np.linalg.inv(self.matrix)
        else:
            bounds = {
                'pillow-ycbcr': ([0, 0, 0], [1, 1, 1]),
                'ycocg-r': ([0, -1, -1], [1, 1, 1]),
                'rct': ([0, -1, -1], [1, 1, 1]),
                'xyz': ([0, 0, 0], WHITE),
                'lab': ([0, -128, -128], [100, 128, 128]),
                'oklab': ([0, -.5, -.5], [1, .5, .5]),
                'ictcp': ([0, -1, -1], [1, 1, 1]),
                'hsv': ([0, 0, 0], [1, 1, 1]),
                'hsl': ([0, 0, 0], [1, 1, 1]),
                'gray': ([0, 0, 0], [1, 1, 1]),
            }
            if self.base == 'pca':
                raise ValueError('PCA requires a fitted matrix and mean')
            self.low, self.high = map(np.asarray, bounds[self.base])

    @classmethod
    def fit_pca(cls, name, pixels):
        pixels = np.asarray(pixels, float).reshape(-1, 3)
        if name.endswith('-linear'):
            pixels = linearize(pixels)
        mean = pixels.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov(pixels-mean, rowvar=False))
        matrix = vectors[:, ::-1].T.copy()
        for row in matrix:
            if row[np.argmax(np.abs(row))] < 0:
                row *= -1
        return cls(name, matrix, mean)

    def record(self):
        return {'name': self.name, 'matrix': None if self.matrix is None else
                self.matrix.tolist(), 'mean': None if self.mean is None else
                self.mean.tolist(), 'low': self.low.tolist(),
                'high': self.high.tolist(),
                'ictcp_sdr_white_nits': 100 if self.base == 'ictcp' else None}

    @classmethod
    def from_record(cls, record):
        return cls(record['name'], record.get('matrix'), record.get('mean'))

    def forward(self, rgb):
        x = np.asarray(rgb, float)
        if self.matrix is not None:
            work = linearize(x) if self.linear else x
            raw = (work-self.mean)@self.matrix.T
        elif self.base == 'pillow-ycbcr':
            raw = np.asarray(Image.fromarray(np.uint8(np.clip(
                x*255, 0, 255))).convert('YCbCr'), float)/255
        elif self.base in ('ycocg-r', 'rct'):
            r, g, b = np.moveaxis(np.rint(x*255), -1, 0)
            if self.base == 'rct':
                raw = np.stack((np.floor((r+2*g+b)/4), b-g, r-g), -1)/255
            else:
                co = r-b
                t = b+np.floor(co/2)
                cg = g-t
                raw = np.stack((t+np.floor(cg/2), co, cg), -1)/255
        elif self.base in ('xyz', 'lab'):
            raw = linearize(x)@XYZ.T
            if self.base == 'lab':
                t = raw/WHITE
                f = np.where(t > (6/29)**3, np.cbrt(t), t/(3*(6/29)**2)+4/29)
                raw = np.stack((116*f[..., 1]-16,
                                500*(f[..., 0]-f[..., 1]),
                                200*(f[..., 1]-f[..., 2])), -1)
        elif self.base == 'oklab':
            raw = np.cbrt(linearize(x)@OK_M1.T)@OK_M2.T
        elif self.base == 'ictcp':
            bt2020 = linearize(x)@XYZ.T@np.linalg.inv(RGB2020).T
            raw = _pq(bt2020@ICT_LMS.T*.01)@ICT_M.T
        elif self.base in ('hsv', 'hsl'):
            high, low = x.max(-1), x.min(-1)
            delta = high-low
            r, g, b = np.moveaxis(x, -1, 0)
            safe = np.maximum(delta, 1e-12)
            hue = np.select([high == r, high == g],
                            [(g-b)/safe, (b-r)/safe+2], default=(r-g)/safe+4)/6 % 1
            hue = np.where(delta > 0, hue, 0)
            value = high if self.base == 'hsv' else (high+low)/2
            denominator = high if self.base == 'hsv' else 1-np.abs(2*value-1)
            saturation = np.where(delta > 0, delta/np.maximum(denominator, 1e-12), 0)
            raw = np.stack((hue, saturation, value), -1)
        else:
            y = x@np.array([.2126, .7152, .0722])
            raw = np.stack((y, y*0, y*0), -1)
        return (raw-self.low)/(self.high-self.low)

    def inverse(self, planes):
        raw = np.asarray(planes, float)*(self.high-self.low)+self.low
        if self.matrix is not None:
            rgb = raw@self.inverse_matrix.T+self.mean
            return delinearize(rgb) if self.linear else rgb
        if self.base == 'pillow-ycbcr':
            y, cb, cr = np.moveaxis(raw, -1, 0)
            cb, cr = cb-128/255, cr-128/255
            return np.stack((y+1.402*cr, y-.344136*cb-.714136*cr,
                             y+1.772*cb), -1)
        if self.base in ('ycocg-r', 'rct'):
            # Integer lifting remains reversible at exact recovered values.
            # With wire noise its defined inverse still uses floor lifting.
            y, c1, c2 = np.moveaxis(raw*255, -1, 0)
            if self.base == 'rct':
                g = y-np.floor((c1+c2)/4+1e-10)
                rgb = np.stack((c2+g, g, c1+g), -1)
            else:
                t = y-np.floor(c2/2+1e-10)
                g = c2+t
                b = t-np.floor(c1/2+1e-10)
                rgb = np.stack((c1+b, g, b), -1)
            return rgb/255
        if self.base in ('xyz', 'lab'):
            xyz = raw
            if self.base == 'lab':
                l, a, b = np.moveaxis(raw, -1, 0)
                fy = (l+16)/116
                f = np.stack((fy+a/500, fy, fy-b/200), -1)
                xyz = WHITE*np.where(f > 6/29, f**3, 3*(6/29)**2*(f-4/29))
            return delinearize(xyz@np.linalg.inv(XYZ).T)
        if self.base == 'oklab':
            return delinearize((raw@np.linalg.inv(OK_M2).T)**3@np.linalg.inv(OK_M1).T)
        if self.base == 'ictcp':
            lms = _pq(raw@np.linalg.inv(ICT_M).T, inverse=True)/.01
            return delinearize(lms@np.linalg.inv(ICT_LMS).T@RGB2020.T@np.linalg.inv(XYZ).T)
        if self.base in ('hsv', 'hsl'):
            values = raw.copy()
            values[..., 0] %= 1
            values[..., 1:] = np.clip(values[..., 1:], 0, 1)
            h, s, v = np.moveaxis(values, -1, 0)
            c = v*s if self.base == 'hsv' else (1-np.abs(2*v-1))*s
            x = c*(1-np.abs((h*6)%2-1))
            sector = np.floor(h*6).astype(int)
            red = np.select([sector == 0, sector == 1, sector == 4, sector == 5],
                            [c, x, x, c], default=0)
            green = np.select([sector == 0, sector == 1, sector == 2, sector == 3],
                              [x, c, c, x], default=0)
            blue = np.select([sector == 2, sector == 3, sector == 4, sector == 5],
                             [x, c, c, x], default=0)
            m = v-c if self.base == 'hsv' else v-c/2
            return np.stack((red+m, green+m, blue+m), -1)
        return np.repeat(raw[..., :1], 3, axis=-1)
