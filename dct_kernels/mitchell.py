"""Mitchell-Netravali cubic: the B/C family (B=0, C=0.5 is Catmull-Rom)."""
import numpy as np

LABEL = 'Mitchell-Netravali'
HELP = ('Cubic family. B softens, C sharpens. B=1/3 C=1/3 is the standard '
        'compromise; B=0 C=0.5 is Catmull-Rom (sharper, a little ringing). '
        'The default C=0.8 is a sharper setting.')
SUPPORT = 2.0
PARAMS = {
    'B': (0.33, 0.0, 1.0, 0.05, 'blur'),
    'C': (0.8, 0.0, 1.0, 0.05, 'sharpening'),
}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {'B': 0.0, 'luma_mix': 0.25},
    'aspect-fold-500': {'luma_mix': 0.05},
}


def kernel(x, B, C):
    t = np.abs(np.asarray(x, np.float64))
    near = ((12 - 9*B - 6*C)*t**3 + (-18 + 12*B + 6*C)*t**2 + (6 - 2*B)) / 6
    far = ((-B - 6*C)*t**3 + (6*B + 30*C)*t**2 + (-12*B - 48*C)*t
           + (8*B + 24*C)) / 6
    return np.where(t < 1, near, np.where(t < 2, far, 0.0))
