"""Codec engine registry: pick the encode/decode engine by name.

Each engine is a black box with the same contract as ``transport3``: an
``encode`` that turns source coefficients into one reference-geometry audio
frame, and a ``Receiver`` that turns audio back into ``Decoded`` results. v3 is
the DCT source coder; v4 swaps the transform for a 2-D wavelet; v5 uses
2-level CDF 9/7 DWT with repeat-protected values on the HD wire layout. Everything else
-- the OFDM wire, the header, the preamble, the acquisition -- is shared and
lives in ``transport3``/``core``, so today the engines differ only in which
``SourceCoder`` subclass they build. The interface is deliberately wider than
that so a future revision can replace the modulation wholesale without the CLI
having to know.
"""
import numpy as np

from . import transport3 as V3
from . import imaging
from .core import SourceCoder, PROFILE_CODES, profile_code, profile_name, V5_PROFILE
from .wavelet import WaveletCoder, Cdf97Coder

__all__ = ['ENGINES', 'get_engine', 'engine_names', 'coder_for', 'coders_for',
           'Engine']


def coder_for(profile, layout=None):
    """Build the right coder (DCT or wavelet) for a profile name."""
    shapes = imaging.plane_shapes(profile)
    if profile == V5_PROFILE:
        # hd-dwt is CDF 9/7 with repeat copies, NOT SourceCoder DCT, and both
        # ends must build it identically or the wire cannot round trip:
        # wavelet.hd_dwt_coder() is the one constructor.
        from .wavelet import hd_dwt_coder
        coder = hd_dwt_coder()
        return coder, coder.shapes
    if profile == 'tape-80x60':
        from .wavelet import tape_80x60_coder
        coder = tape_80x60_coder()
        return coder, coder.grids
    grids = imaging.plane_grids(profile)
    if layout is not None and \
            sum(int(np.prod(s)) for s in shapes) > layout.capacity:
        shapes = imaging.fit_shapes(shapes, layout.capacity)
        grids = shapes
    cls = WaveletCoder if 'wavelet' in profile else SourceCoder
    return cls(shapes, grids=grids), grids


def coders_for(layout):
    """code -> coder for every profile the header may name, across all engines.

    This is what lets one receiver auto-detect v3 or v4 from the header instead
    of being told out of band -- the transition property "do both on decode".
    """
    return {profile_code(name): coder_for(name, layout)[0]
            for name in PROFILE_CODES}


class Engine:
    """One codec engine: encode + decode + the profiles it names on the wire."""

    def __init__(self, name, profiles):
        self.name = name
        self.profiles = tuple(profiles)

    @property
    def wire(self):
        return V3.WIRE

    @property
    def reference_rate(self):
        return V3.REFERENCE_RATE

    def coder_for(self, profile, layout=None):
        if profile not in self.profiles:
            raise ValueError(f'{profile!r} is not a {self.name} profile')
        return coder_for(profile, layout)

    def coders_for(self, layout):
        return {profile_code(name): self.coder_for(name, layout)[0]
                for name in self.profiles}

    def profile_code(self, name):
        return profile_code(name)

    def profile_name(self, code):
        return profile_name(code)

    def encode(self, values, coder, absolute, index, count, stamp_ms=0, flags=0,
               headroom=.95, profile=0, aspect_code=0):
        return V3.encode(values, self.wire, coder, absolute, index, count,
                         stamp_ms=stamp_ms, flags=flags, headroom=headroom,
                          profile=profile, aspect_code=aspect_code)

    def receiver(self, layout=None, coder=None, **kwargs):
        return V3.Receiver(layout or self.wire, coder, **kwargs)

    def adapt(self, audio, rate):
        return V3.band_limited(audio, rate)

    def emit_length(self, samples, rate):
        return V3.emit_length(samples, rate)

    def describe(self):
        return self.wire.describe(self.reference_rate)


class V5Engine(Engine):
    """v5: CDF 9/7 DWT on the HD wire layout, strongest values sent twice.

    Uses the standard V3 wire format and the standard V3.Receiver (edge
    pulse acquisition, OFDM demod) with the matched CDF 9/7 coder and a
    profile=2 header -- the receiver auto-detects hd-dwt from the header the
    same way it auto-detects v3/v4. The experimental mono/LDPC receiver
    (transport3_v5.ReceiverV5) is parked, not in the live path.
    """

    @property
    def wire(self):
        return V3.WIRE_HD

    @property
    def reference_rate(self):
        return V3.REFERENCE_RATE

    def coder_for(self, profile, layout=None):
        if profile != V5_PROFILE:
            raise ValueError(f'{profile!r} is not a v5 profile')
        # One constructor for every sender, receiver and tool.
        from .wavelet import hd_dwt_coder
        coder = hd_dwt_coder()
        return coder, coder.shapes

    def encode(self, values, coder, absolute, index, count, stamp_ms=0,
               headroom=.95, profile=0, aspect_code=0):
        # Standard v3 wire format; the header carries profile=2 (hd-dwt).
        return V3.encode(values, self.wire, coder, absolute, index, count,
                         stamp_ms=stamp_ms, headroom=headroom,
                          profile=profile or 2, aspect_code=aspect_code)

    def receiver(self, layout=None, coder=None, **kwargs):
        return V3.Receiver(layout or self.wire, coder, **kwargs)

    def describe(self):
        return f"{self.wire.describe(self.reference_rate)} (v5: CDF 9/7, profile=2 header)"


ENGINES = {
    'v3': Engine('v3', ('color-dct',)),
    'v4': Engine('v4', ('color-wavelet',)),
    'v5': V5Engine('v5', (V5_PROFILE,)),
}


def engine_names():
    return tuple(ENGINES)


def get_engine(name):
    try:
        return ENGINES[name]
    except KeyError:
        raise KeyError(f'Unknown engine {name!r}; known: {", ".join(ENGINES)}') from None
