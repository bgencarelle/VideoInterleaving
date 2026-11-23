"""2D Contrast Sensitivity Function (CSF) with oblique-effect diamond tuning.

Boosts horizontal and vertical mid-band frequencies where human visual contrast
sensitivity is sharpest, while tapering diagonal frequencies where human eyes
have lower acuity. Reduces diagonal mesh and ringing while increasing cardinal
edge acutance.
"""
import numpy as np

LABEL = 'CSF Diamond'
DEFER_GAIN_TO_CODEC = True
HELP = ('2D Contrast Sensitivity Function with oblique-effect diamond tuning. '
        'Boosts horizontal and vertical frequencies where human vision is most '
        'sensitive, while gently attenuating diagonal frequencies. amount is '
        'the mid-band boost; oblique is diagonal retention (1.0 = circular); '
        'p_norm shapes the diamond contour (< 2.0 = diamond); rolloff is the '
        'high-frequency knee. The default luma mix is 0.25; chroma shaping '
        'remains enabled by default.')
RADIAL = False
HOST_DEFAULTS = {'luma_mix': 0.25}

PARAMS = {
    'amount': (0.05, 0.0, 0.6, 0.05, 'mid-band cardinal boost (0.05 = +5%)'),
    'oblique': (0.95, 0.4, 1.0, 0.05, 'diagonal retention factor (1.0 = round)'),
    'p_norm': (1.9, 1.2, 2.0, 0.1, 'L_p contour power (2.0 = round, 1.0 = diamond)'),
    'centre': (0.18, 0.05, 0.35, 0.01, 'peak frequency (cycles/pixel)'),
    'width': (0.10, 0.03, 0.25, 0.01, 'peak bandwidth'),
    'rolloff': (0.55, 0.25, 0.55, 0.01, 'high-frequency roll-off start'),
    'color_planes': (1, 0, 1, 1, '0 = leave chroma planes unchanged', True),
}
PROFILE_DEFAULTS = {
    'aspect-mono-500': {'luma_mix': 0.1},
    'aspect-fold-500': {'luma_mix': 0.05},
}


def gain(ctx, amount, oblique, p_norm, centre, width, rolloff, color_planes):
    # Chroma is carried at lower resolution and has much lower perceptual
    # acuity. Avoid running a full transform for a window that is not needed.
    if ctx.plane and not color_planes:
        return None
    nu_y, nu_x = ctx.nu()

    # 2D Angle in spatial frequency domain
    theta = np.arctan2(nu_y, nu_x)

    # Oblique effect: 1.0 on horizontal/vertical axes, down to `oblique` at 45 deg
    # cos(4 * theta) is +1 at 0, pi/2; -1 at pi/4
    oblique_factor = ((1.0-oblique)/2.0*np.cos(4.0*theta) +
                      (1.0+oblique)/2.0)

    # L_p frequency distance
    nu_p = (nu_y**p_norm + nu_x**p_norm)**(1.0/p_norm)

    # CSF mid-band Gaussian bump
    bump = amount*np.exp(-0.5*((nu_p-centre)/width)**2)*oblique_factor

    # High-frequency roll-off (cosine taper above rolloff)
    # Effective frequency scaled by oblique factor: diagonals roll off earlier
    eff_nu = nu_p/np.maximum(oblique_factor, 0.1)

    taper = np.where(
        eff_nu > rolloff,
        0.5 + 0.5*np.cos(np.pi*np.clip(
            (eff_nu-rolloff)/(0.55-rolloff+1e-6), 0.0, 1.0)),
        1.0
    )

    # Combine: 1.0 at DC (forced), peak in mid-band, gentle taper at the rim
    g = (1.0 + bump) * np.maximum(taper, 0.0)

    # Normalize DC exactly to 1.0
    g[0, 0] = 1.0
    return g.astype(np.float64)
