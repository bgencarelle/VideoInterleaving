"""Project directly from source-resolution DCT coefficients."""

LABEL = 'Native-source DCT'
HELP = ('Skip the intermediate block-average and 2:1 decimation stages. The '
        'original-resolution frame is projected directly onto the existing '
        'coder grid and transmitted band with Numba-compiled source conversion '
        'and partial DCT; the wire format is unchanged. The full-source path '
        'uses more CPU and can lower live FPS.')


def prefilter():
    return {'full_source': True}


# Nothing to tune: every folded profile runs it as it is. (Without entries
# the sender refused it on every profile.)
PROFILE_DEFAULTS = {name: {} for name in (
    'fold-500', 'mono-colour-500', 'aspect-fold-500', 'aspect-mono-500',
    'stereo-slices', 'aspect-mono-nested', 'stereo-nested',
    'adaptive-mono-nested', 'adaptive-stereo-nested')}


# Nothing to tune: every folded profile runs it as it is. (Without entries
# the sender refused it on every profile.)
PROFILE_DEFAULTS = {name: {} for name in (
    'fold-500', 'mono-colour-500', 'aspect-fold-500', 'aspect-mono-500',
    'stereo-slices', 'aspect-mono-nested', 'stereo-nested',
    'adaptive-mono-nested', 'adaptive-stereo-nested')}


# Nothing to tune: every folded profile runs it as it is. (Without entries
# the sender refused it on every profile.)
PROFILE_DEFAULTS = {name: {} for name in (
    'fold-500', 'mono-colour-500', 'aspect-fold-500', 'aspect-mono-500',
    'stereo-slices', 'aspect-mono-nested', 'stereo-nested',
    'adaptive-mono-nested', 'adaptive-stereo-nested')}


# Nothing to tune: every folded profile runs it as it is. (Without entries
# the sender refused it on every profile.)
PROFILE_DEFAULTS = {name: {} for name in (
    'fold-500', 'mono-colour-500', 'aspect-fold-500', 'aspect-mono-500',
    'stereo-slices', 'aspect-mono-nested', 'stereo-nested',
    'adaptive-mono-nested', 'adaptive-stereo-nested')}
