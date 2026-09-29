"""V7 production and historical wire profiles for offline benches.

The application sender defaults to the M=500 luma fold with coded pilots and
the end-of-packet (EOF) marker (``modem_v7_display.py``). The low-level encoder's
defaults are different -- no pilot tones, no EOF marker, no fold -- so a bench
that calls ``v7.encode_pulse_stream`` directly measures a wire the application
does not send.
``WireProfile('default')`` uses the application's production packet encoder
(``modem_v7_display.encode_values_packet``) for folded packets and installs the
standalone receiver hooks for offline decoding.

A tool that derives its own model (other grids or coefficient shapes) cannot
fold: the pinned fold tables belong to the canonical box model. It uses
``WireProfile('default', fold=False)``, which keeps the live framing (EOF
marker, coded pilots carrying a fold-off status, tone-seeded timing, EOF
boundaries) without the fold.

``WireProfile('baseline')`` is the historical nearest / no-tone / no-EOF wire
with next-header boundaries. It exists for comparison with older results and
for tests of the low-level defaults; it is not what the senders emit.
"""
from contextlib import ExitStack, contextmanager

import numpy as np

from animation_modem import v7
from tools import v7_live

PROFILES = ('default', 'baseline')
FOLD_SLOTS = 500
# What tools/v7_live.py's receiver uses for the default profile.
DEFAULT_DECODE = {'pilot_timing': 'tone-seeded', 'frame_boundary': 'eof'}


class WireProfile:
    def __init__(self, name='default', fold=True):
        if name not in PROFILES:
            raise ValueError(f'profile must be one of {PROFILES}')
        self.name = name
        self.folded = bool(fold) and name == 'default'
        self.encode_filter = 'box' if self.folded else 'nearest'
        self._fold = None
        self._production_fold = None
        if name == 'default':
            v7_live._ensure_test_modem_path()
            import tone_code
            self._tone = tone_code
            if self.folded:
                from live_fold import LiveFold
                self._fold = LiveFold(FOLD_SLOTS)
            self._status_mode = (tone_code.FOLD_500 if self.folded else
                                 tone_code.FOLD_OFF)
            self._status = tone_code.encode_status(self._status_mode)

    @property
    def label(self):
        if self.name == 'baseline':
            return 'baseline (nearest, no tones, no EOF)'
        return ('default (fold 500, coded pilots, EOF)' if self.folded else
                'default framing, fold off (coded pilots, EOF)')

    def check(self, model):
        """Fail before encoding if a folded profile is given another model."""
        if self._fold is not None:
            self._fold.check(model)

    # ------------------------------------------------------------ encode
    def encode(self, model, values, start_counter=1, aspect_codes=None,
               source_indices=None, eof_marker=False, loop=None,
               directions=None):
        """48 kHz stereo packets for ``values`` (one vector per packet).

        The default profile always carries the EOF marker. ``eof_marker``
        only adds it to the baseline wire (reverse playback needs it).
        """
        values = list(values)
        if self.name == 'baseline':
            return v7.encode_pulse_stream(model, values, start_counter,
                                          aspect_codes, source_indices, loop,
                                          directions,
                                          eof_marker=bool(eof_marker))
        self.check(model)
        codes = list(aspect_codes) if aspect_codes is not None else [0]*len(values)
        indexes = (list(source_indices) if source_indices is not None else
                   [start_counter+i-1 for i in range(len(values))])
        ways = list(directions) if directions is not None else [1]*len(values)
        if not len(codes) == len(indexes) == len(ways) == len(values):
            raise ValueError(
                'aspect_codes, source_indices, and directions must match values')
        if self._fold is not None:
            from animation_modem.v7_fold import Fold500
            from modem_v7_display import encode_values_packet
            if self._production_fold is None:
                self._production_fold = Fold500(model)
            return np.concatenate([
                encode_values_packet(
                    model, value, start_counter+offset, indexes[offset],
                    codes[offset], loop=loop, direction=ways[offset],
                    eof_marker=True, fold=self._production_fold)
                for offset, value in enumerate(values)])

        packets = []
        for offset, (value, code, index, way) in enumerate(
                zip(values, codes, indexes, ways)):
            counter = start_counter+offset
            packet = v7.encode_pulse_frame(
                model, value, counter, aspect_code=code, source_index=index,
                loop=loop, direction=way, pilot_tones=False, eof_marker=True,
                pulse_profile_code=self._status_mode)
            packets.append(self._tone.add_tone_code(
                packet, counter, self._status))
        return np.concatenate(packets)

    # -------------------------------------------------------------- decode
    @property
    def decode_options(self):
        """Receiver settings for this profile's wire (a fresh dict)."""
        return dict(DEFAULT_DECODE) if self.name == 'default' else {}

    @contextmanager
    def receiving(self):
        """Install the fold's equaliser hooks and the coded-pilot timing hook
        for the block only (default profile); the baseline needs none."""
        with ExitStack() as stack:
            if self.name == 'default':
                if self._fold is not None:
                    self._fold.install()
                    stack.callback(self._fold.uninstall)
                stack.enter_context(self._tone.coded_pilot_timing())
            yield

    def decode(self, model, audio, **kwargs):
        """``v7.decode_pulse_stream`` with this profile's receiver settings."""
        options = self.decode_options
        options.update(kwargs)
        with self.receiving():
            return v7.decode_pulse_stream(model, audio, **options)

    def values(self, model, result):
        """Display values for one decoded packet (unfolded when folded)."""
        if self._fold is None:
            return v7.values_from(model, result.coeffs)
        return self._fold.values(
            model, result,
            metadata_confirmed=v7_live._coded_mode_matches_fold(result, self._fold))


def add_profile_argument(parser):
    parser.add_argument(
        '--profile', choices=PROFILES, default='default',
        help='wire to test: the application default (fold 500 where the model allows, '
             'coded pilots, EOF marker) or the historical baseline (nearest, no '
             'tones, no EOF; default: default)')
