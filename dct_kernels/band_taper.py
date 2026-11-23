"""A smooth window over the sent band, defined from its centre outwards."""
import numpy as np

LABEL = 'Band taper'
HELP = ('Radial raised-cosine over the sent band. edge is the gain left at '
        'the rim of the band (1 = no window); power moves the knee outwards '
        '(>1 keeps the middle flat) or inwards. Reduces ringing and the '
        'mesh near edges without a visible blur at moderate settings.')
RADIAL = False
PARAMS = {
    'edge': (0.5, 0.0, 1.0, 0.05, 'gain at the rim of the band'),
    'power': (3.0, 0.5, 4.0, 0.1, 'knee position'),
    'use_guests': (0, 0, 1, 1, '1 = measure the band to the folded guests', True),
}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {
        'edge': 0.8,
        'luma_mix': 0.25,
        'use_guests': 1,
    },
    'aspect-fold-500': {
        'edge': 0.9,
        'luma_mix': 0.05,
        'use_guests': 1,
    },
}


def gain(ctx, edge, power, use_guests):
    if use_guests and ctx.extent != ctx.sent:
        radius = ctx.band(extent=True)
    else:
        radius = ctx.band()
    r = np.clip(radius, 0.0, 1.0) ** power
    return edge + (1.0 - edge) * (0.5 + 0.5*np.cos(np.pi*r))
