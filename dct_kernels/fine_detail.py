"""Average the source in smaller, finer steps before the transform."""
LABEL = 'Fine-detail pre-shrink'
HELP = ('Before the DCT the frame is averaged down to a plane a few times '
        'the grid. Whatever is finer than that average folds into the sent '
        'band as aliasing (a screen door, text, fabric). A larger factor '
        'averages less coarsely and so aliases less, at more CPU: 4 is the '
        'shipped encoder, 6 costs about 3 ms per 1080p frame, 8 about 14 ms. '
        'Plain pictures do not change; detailed ones get cleaner in the top '
        'half of the band. It changes only the input of the transform: to '
        'combine it with a window or refit, add a gain or post function to '
        'this file.')
PARAMS = {'factor': (6.0, 1.75, 8.0, 0.25, 'plane size as a multiple of the grid')}


def prefilter(factor):
    return {'preshrink': float(factor)}
