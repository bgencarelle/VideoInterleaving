"""Shared pieces of the test_modem_v7 picture-quality experiments.

Frames: any set of large source frames (the experiments were run on 810x1080
portrait video frames). Frames 1, 3, 5, ... fit the statistics and frames
2, 4, 6, ... are scored, so no frame is scored against statistics fitted on
itself.

Scoring: every reconstruction is resized to DISPLAY and compared with the
source resized to DISPLAY, which is how a receiver shows the picture.
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from animation_modem import v7                                         # noqa: E402
from animation_modem.imaging import values_image                        # noqa: E402
from tools.v7_torture_matrix import CASES, RATE, TARGET, impair         # noqa: E402,F401

try:
    from ssimulacra2 import compute_ssimulacra2
except ImportError:                                                     # pragma: no cover
    sys.exit('test_modem_v7 needs the ssimulacra2 package: pip install ssimulacra2')
from skimage.color import deltaE_ciede2000, rgb2lab                     # noqa: E402

DISPLAY = (405, 540)
STEADY_FROM = v7.TAIL_PHASES + 1        # first packet with every tail slice received
BOX = Image.Resampling.BOX


def load_frames(paths):
    frames = [Image.open(path).convert('RGB') for path in paths]
    if len(frames) < 2:
        sys.exit('give at least two frames: even ones fit statistics, odd ones are scored')
    return frames[0::2], frames[1::2]


def reference(frame):
    return frame.resize(DISPLAY, Image.Resampling.LANCZOS)


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    return buffer


def ssimulacra2(ref, got):
    return float(compute_ssimulacra2(_png(ref), _png(got)))


def score(ref, got):
    """SSIMULACRA2 (higher is better), luma PSNR and mean CIEDE2000."""
    ry = np.asarray(ref.convert('YCbCr'))[..., 0].astype(float)
    gy = np.asarray(got.convert('YCbCr'))[..., 0].astype(float)
    return {'ssimulacra2': ssimulacra2(ref, got),
            'psnr_y': float(10*np.log10(255**2/np.mean((ry-gy)**2))),
            'de2000': float(np.mean(deltaE_ciede2000(rgb2lab(np.asarray(ref)),
                                                     rgb2lab(np.asarray(got)))))}


class Grids:
    """Orthonormal DCT of every plane of a value vector on `grids`."""

    def __init__(self, grids):
        self.grids = tuple(tuple(g) for g in grids)
        self.off = np.cumsum([0] + [r*c for r, c in self.grids])
        self.plane = np.concatenate([np.full(r*c, p) for p, (r, c) in enumerate(self.grids)])
        self.rad = np.concatenate([
            np.hypot(np.arange(r)[:, None]/r, np.arange(c)[None, :]/c).ravel()
            for r, c in self.grids])

    def forward(self, values):
        return np.concatenate([dctn(values[a:b].reshape(g), norm='ortho').ravel()
                               for a, b, g in zip(self.off, self.off[1:], self.grids)])

    def inverse(self, coeffs):
        return np.concatenate([idctn(coeffs[a:b].reshape(g), norm='ortho').ravel()
                               for a, b, g in zip(self.off, self.off[1:], self.grids)])

    def picture(self, coeffs):
        return values_image(self.inverse(coeffs), self.grids).resize(
            DISPLAY, Image.Resampling.BICUBIC)

    def corner_positions(self, shapes):
        """Grid position of every coefficient of the per-plane corners `shapes`,
        in the V7 coder's order (plane by plane, each corner row-major)."""
        return np.concatenate([
            self.off[p] + (np.arange(r)[:, None]*gc + np.arange(c)[None, :]).ravel()
            for p, ((r, c), (gr, gc)) in enumerate(zip(shapes, self.grids))])


def fit_variances(grid, sample, frames, crops=40, seed=1):
    """Per-coefficient mean and variance: the V7 derive_tables method (random
    crops, flips, a radial power law per plane) on the given frames."""
    rng = np.random.default_rng(seed)
    C = []
    for image in frames:
        W, H = image.size
        for _ in range(crops):
            s = rng.uniform(.6, 1.0); w = int(W*s); h = int(w*4/3)
            x = rng.integers(0, W-w+1); y = rng.integers(0, H-h+1)
            crop = image.crop((x, y, x+w, y+h))
            if rng.random() < .5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            C.append(grid.forward(sample(crop)))
    C = np.asarray(C)
    mu, lam = np.zeros(C.shape[1]), np.empty(C.shape[1])
    for p in range(len(grid.grids)):
        sl = slice(grid.off[p], grid.off[p+1])
        mu[grid.off[p]] = C[:, grid.off[p]].mean()
        L = np.mean(C[:, sl]**2, axis=0)
        rad = grid.rad[sl]; m = rad > 0
        law = lambda R, a, k, q: a - q*np.log1p(k*R)
        (a, k, q), _ = curve_fit(law, rad[m], np.log(np.maximum(L[m], 1e-12)), p0=(0, 20, 2),
                                 bounds=([-50, 0, 0], [50, 1e4, 10]), maxfev=20000)
        fit = np.exp(law(rad, a, k, q)); fit[0] = C[:, grid.off[p]].var() + 1e-6
        lam[sl] = fit
    return mu, lam


def v7_values(image, encode_filter):
    """What the V7 sender transmits for `image` with this encode filter."""
    return v7.image_values(v7.prepare_image(image, encode_filter), v7.V7_GRIDS, encode_filter)


def box_values(image, grids):
    """Box-sample the source straight onto `grids` (luma), chroma box-reduced."""
    rows, cols = grids[0]
    planes = image.resize((cols, rows), BOX).convert('YCbCr').split()
    return np.concatenate([np.asarray(plane.resize((c, r), BOX)).ravel()
                           for plane, (r, c) in zip(planes, grids)]).astype(float)/127.5 - 1


def jitter_warp(x, rms=.001, band=(20, 300), seed=11):
    """Random tape-speed error (scrape flutter / capstan jitter), band-limited,
    RMS as a fraction of speed, applied to audio at RATE."""
    rng = np.random.default_rng(seed)
    n = sosfilt(butter(2, band, btype='bandpass', fs=RATE, output='sos'),
                rng.standard_normal(len(x)))
    n *= rms/np.sqrt(np.mean(n*n))
    position = np.arange(len(x)) + np.cumsum(n)
    base = np.arange(len(x), dtype=float)
    return np.column_stack([np.interp(position, base, x[:, ch])
                            for ch in range(x.shape[1])]).astype(np.float32)


def contact_sheet(path, items, columns=3):
    """items: [(label, 405x540 image)] -> one labelled PNG."""
    from PIL import ImageDraw
    W, H = DISPLAY
    pad = 18
    rows = (len(items) + columns - 1)//columns
    sheet = Image.new('RGB', (W*columns, (H+pad)*rows), 'white')
    draw = ImageDraw.Draw(sheet)
    for k, (label, image) in enumerate(items):
        x, y = (k % columns)*W, (k//columns)*(H+pad)
        sheet.paste(image, (x, y+pad))
        draw.text((x+6, y+3), label, fill='black')
    sheet.save(path)
