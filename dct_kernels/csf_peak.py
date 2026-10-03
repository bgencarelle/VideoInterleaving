"""Boost the middle of the band, where a blown-up picture is read."""
import numpy as np

LABEL = 'Mid-band emphasis'
HELP = ('Contrast sensitivity peaks at middle spatial frequencies, and a '
        'picture that is enlarged is read mostly through them. A smooth '
        'bump at `centre` (cycles per sent pixel) lifts that band and a '
        'gentle fall-off above `rolloff` keeps the top of the band from '
        'ringing. centre 0.1 is blobs and shapes; 0.3 is texture and edges.')
PARAMS = {
    'centre': (0.18, 0.03, 0.5, 0.01, 'bump position, cycles per sent pixel'),
    'width': (0.09, 0.02, 0.4, 0.01, 'bump width'),
    'amount': (0.15, 0.0, 2.0, 0.05, 'how much the bump adds (0.15 = +15%)'),
    'rolloff': (0.38, 0.1, 1.0, 0.01, 'frequency where the top starts to fall'),
}


def response(nu, centre, width, amount, rolloff):
    nu = np.asarray(nu, float)
    bump = 1.0 + amount*np.exp(-0.5*((nu - centre)/width)**2)
    bump = bump/(1.0 + amount*np.exp(-0.5*(centre/width)**2))      # 1 at DC
    fall = np.where(nu > rolloff,
                    0.5 + 0.5*np.cos(np.pi*np.clip((nu - rolloff)/(0.5 - rolloff + 1e-6), 0, 1)),
                    1.0)
    return bump*np.maximum(fall, 0.0) + 0.0*nu
