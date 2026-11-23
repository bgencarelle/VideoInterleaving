"""Standalone live V7 sender/receiver profile names and signalling."""

from animation_modem.transport3 import (ADAPTIVE_MONO_MODE,
                                        ADAPTIVE_STEREO_MODE)
ADAPTIVE_NESTED_MODE = ADAPTIVE_MONO_MODE

# Human-facing profiles. Older wire identities remain decode-only except for
# the explicitly retained aspect-compatible choices.
SEND_PROFILES = (
    ('Stereo · Aspect Fold 500 · recommended', 'aspect-fold-500'),
    ('Mono · Aspect Fold 500 · legacy-compatible', 'aspect-mono-500'),
    ('Mono · Legacy nested', 'aspect-mono-nested'),
    ('Stereo · Legacy nested', 'stereo-nested'),
    ('Mono · Adaptive nested', 'adaptive-mono-nested'),
    ('Stereo · Adaptive nested', 'adaptive-stereo-nested'),
)

ADAPTIVE_PROFILES = {
    'adaptive-mono-nested': {'mode': ADAPTIVE_MONO_MODE, 'family': 'mono'},
    'adaptive-stereo-nested': {'mode': ADAPTIVE_STEREO_MODE, 'family': 'stereo'},
}
