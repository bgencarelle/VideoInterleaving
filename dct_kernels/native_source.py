"""Project directly from source-resolution DCT coefficients."""

LABEL = 'Native-source DCT'
HELP = ('Skip the intermediate block-average and 2:1 decimation stages. The '
        'original-resolution frame is projected directly onto the existing '
        'coder grid and transmitted band with Numba-compiled source conversion '
        'and partial DCT; the wire format is unchanged. The full-source path '
        'uses more CPU and can lower live FPS.')


def prefilter():
    return {'full_source': True}
