"""The live sender's wires and the live receiver's profile dispatcher, in
memory, for the nested-fold unit tests.

Not a picture-quality judge: pictures are judged only through the live
sender and receiver (docs/V7_FIXES.md, testing rules). This used to be part
of tools/v7_nested_eval.py, which was retired with fixes list items 11-13.
"""
import io
from contextlib import redirect_stdout

import numpy as np
from scipy.signal import resample_poly

from animation_modem import v7
from tools import v7_live

v7_live._ensure_test_modem_path()
import nested_fold                                                      # noqa: E402
import tone_code                                                        # noqa: E402
from aspect_fold import AspectFoldWire                                  # noqa: E402
from aspect_mono import AspectMonoWire                                  # noqa: E402

TARGET = .1521/np.sqrt(1+10**(v7.CLOCK_REL_DB/10))


class Rig:
    """Live sender wires and the live receiver's dispatcher, in memory."""

    def __init__(self, layout='auto'):
        self.base = v7.load_model(TARGET, 'box')
        self.dispatcher = v7_live._AdaptiveProfileDecoder(
            v7_live._experimental_fold(500), self.base)
        self.senders = {
            'aspect-mono-500': AspectMonoWire(self.base, side='left'),
            'mono nested': nested_fold.enable_mono(AspectMonoWire(self.base, side='left'), send=True),
            'stereo-slices': nested_fold.slice_wire('auto'),
            'stereo nested': nested_fold.slice_wire('auto', send=True),
            'aspect-fold-500': AspectFoldWire('auto', v7_live.ASPECT_FOLD_TAIL)}

    # ----------------------------------------------------------------- send
    def _values(self, rgb, luma_mask, chroma_masks):
        """Grid values as the live sender makes them for this wire."""
        return v7_live._values(self.base, rgb, 'box', 1.0, dct_encode=True,
                               chroma_sent_for=chroma_masks)[:2]

    def audio(self, mode, rgb, packets):
        """48 kHz stereo stream of ``packets`` packets of one picture."""
        wire = self.senders[mode]
        if mode in ('aspect-mono-500', 'mono nested'):
            values, code = self._values(rgb, None, lambda c: v7_live._chroma_sent_masks(
                wire._codec(wire._packet_model(self.base, c))))
            return np.concatenate([
                wire.encode_packet(self.base, values, counter, aspect_code=code,
                                   source_index=counter-1)
                for counter in range(1, packets+1)]).astype(np.float64)
        if mode in ('stereo-slices', 'stereo nested'):
            for code in range(8):
                wire.model_for(self.base, wire.layout_for(code))
            values, code = self._values(rgb, None, lambda c: wire.chroma_sent_masks(wire.layout_for(c)))
            return wire.encode(self.base, [values]*packets,
                               aspect_codes=[code]*packets).astype(np.float64)
        values, code = self._values(rgb, None, lambda c: v7_live._chroma_sent_masks(
            wire.codec(wire.model_for(self.base, wire.layout_for(c)))))
        model, coefficients = wire.encode_coefficients(self.base, values, code)
        return np.concatenate([
            tone_code.add_tone_code(
                v7.encode_pulse_frame_coeffs(
                    model, coefficients, counter, aspect_code=code,
                    source_index=counter-1, pilot_tones=False,
                    pulse_profile_code=wire.pulse_profile_code),
                counter, tone_code.encode_status(wire.status_mode))
            for counter in range(1, packets+1)]).astype(np.float64)

    # -------------------------------------------------------------- receive
    def _results(self, audio, rate, mode, single, reverse):
        """Decoded results of one input (stereo, or one channel as mono)."""
        dispatcher = self.dispatcher
        audio = np.ascontiguousarray(audio[::-1] if reverse else audio, dtype=np.float32)
        dispatcher.active_mode = dispatcher.dispatch_mode = mode
        if single:
            dispatcher.active_side = 0
        state = v7.PulseState(tail_memory=False)
        if reverse:
            state.set_playback_direction(-1)
            results = []
            for start, scale, _confidence, way in v7.pulse_frame_hits(audio, sample_rate=rate):
                if way < 0:
                    results += v7.decode_reverse_packet(
                        self.base, audio, start, scale, state=state, sample_rate=rate,
                        pilot_timing='tone-seeded')[0]
            return results
        return v7.decode_pulse_stream(self.base, audio, state=state, sample_rate=rate,
                                      pilot_timing='tone-seeded')[0]

    def shown(self, mode, capture, rate=96000, reverse=False, legs=None):
        """[(source index, grid values, note)] for every picture shown.

        ``capture``: what the receiver hears (stereo, or one channel for a
        mono sum).  ``legs``: for the two-channel wires, which input channels
        to read (default: every channel present, forwards or reversed)."""
        dispatcher = self.dispatcher
        dispatcher._last_slices = False
        dispatcher._last_layouts.clear()
        dispatcher.slice_wire.reset()
        dispatcher.aspect_mono_wire.reset_nested()
        capture = np.asarray(capture)
        if capture.ndim == 1:
            capture = capture[:, None]
        channels = capture.shape[1]
        out = []
        dispatcher.install()
        try:
            with tone_code.coded_pilot_timing(), redirect_stdout(io.StringIO()):
                if mode == 'aspect-fold-500':
                    if channels < 2:
                        capture = np.repeat(capture, 2, axis=1)
                    for result in self._results(capture, rate, dispatcher.aspect_mode, False, reverse):
                        self._keep(out, result, lambda r: dispatcher.values(self.base, r))
                elif mode in ('aspect-mono-500', 'mono nested'):
                    # The mono wire is on the left channel; a mono sum has one.
                    for result in self._results(capture[:, :1], rate,
                                                dispatcher.aspect_mono_mode, True, reverse):
                        self._keep(out, result, lambda r: dispatcher.values(self.base, r))
                else:
                    use = list(range(channels)) if legs is None else list(legs)
                    per_leg = {}
                    for leg in use:
                        for result in self._results(capture[:, leg:leg+1], rate,
                                                    dispatcher.aspect_mono_mode, True, reverse):
                            index = result.diag.get('source_index')
                            if index is None or result.status == 'lost':
                                continue
                            half = dispatcher.slice_half(self.base, result, leg)
                            if half is not None:
                                per_leg.setdefault(int(index), []).append((result, half))
                    for index in sorted(per_leg, reverse=reverse):
                        result = per_leg[index][0][0]
                        halves = [half for _, half in per_leg[index]]
                        values = dispatcher.slice_values(halves, result)
                        out.append((index, np.asarray(values, float),
                                    result.diag.get('slices_shown', '')))
        finally:
            dispatcher.uninstall()
        return out

    @staticmethod
    def _keep(out, result, values):
        index = result.diag.get('source_index')
        if index is None or (result.status == 'lost' and not result.diag.get('displayable')):
            return
        out.append((int(index), np.asarray(values(result), float), result.status))


def wide(audio48):
    """The stock matrix's conversion of the 48 kHz wire to a 96 kHz capture."""
    return resample_poly(audio48, 2, 1, axis=0)
