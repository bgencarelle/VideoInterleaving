"""What a V7 sender does when nothing is chosen.

One definition, read by both the command line (tools/v7_live.py send) and the
sender GUI (tools/v7_send_gui.py).  A sender started with no switches and a
GUI nobody has touched hand the modem the same packets; every difference is
an argument somebody gave.
"""

DEFAULT_PROFILE = 'aspect-fold-500'

# Nested-fold modes ride on these wires and share their kernel settings.
NESTED_BASE_PROFILES = {'aspect-mono-nested': 'aspect-mono-500',
                        'stereo-nested': 'stereo-slices'}

# Picture encode: straight from the source to the wire's DCT grid, with the
# luma re-fit.
SENDER_DEFAULTS = {
    'profile': DEFAULT_PROFILE,
    'dct_encode': True,
    'luma_adjust': True,
}

REFERENCE_KERNEL = 'reference'
PROFILE_DEFAULT_KERNELS = {
    'aspect-mono-500': 'viewer_solve',
    'aspect-fold-500': 'viewer_solve',
    # Scored through the live nested sender and receiver at a bilinear
    # display (docs/V7_NESTED_FOLD_ISSUES.md): the mono fold does best with
    # its base wire's kernel, the stereo fold with this one.  The stock
    # stereo-slices wire gains nothing from either and stays on the reference.
    'stereo-nested': 'upscale_precomp',
}


def default_kernel_for_profile(profile):
    """Kernel enabled by default for the selected sender wire profile."""
    if profile in PROFILE_DEFAULT_KERNELS:
        return PROFILE_DEFAULT_KERNELS[profile]
    return PROFILE_DEFAULT_KERNELS.get(
        NESTED_BASE_PROFILES.get(profile, profile), REFERENCE_KERNEL)
