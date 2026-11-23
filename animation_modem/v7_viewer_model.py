"""How the receiver draws a picture, said once.

The receiver's display defaults (tools/v7_gl_viewer.py, used by both the
command-line viewer and the receiver GUI) are read from here.
"""

RECEIVER_DISPLAY = {
    # Sent coefficients are enlarged in frequency space (zero-padded DCT) ...
    'enlarge': 'dct',
    'factor': 4,
    # ... the edge-consistent rebuild is mixed in at this strength ...
    'edge': 'on',
    'edge_strength': .75,
    # ... chroma is rebuilt on the luma grid, guided by brightness ...
    'chroma': 'guided',
    # ... and the shader draws the result to the window with this filter.
    'draw': 'bicubic',
}
