"""Find a stereo modem pair and retain it until the input device is reopened."""
from itertools import combinations

import numpy as np

from .transport2 import Receiver


class AutoChannels:
    def __init__(self, layout, coder, channels, transform_factory=None):
        if channels < 2:
            raise ValueError('Stereo modem reception requires at least two input channels')
        self.layout, self.coder, self.channels = layout, coder, channels
        self.transform_factory = transform_factory
        self.selected = None
        self.reset()

    def reset(self):
        """Discard discontinuous audio, preserving an established channel pair."""
        self.candidates = ({self.selected: self._new()}
                           if self.selected is not None else {})
        self.samples = 0
        self.history = np.empty((0, self.channels), np.float32)

    def _new(self):
        transform = self.transform_factory() if self.transform_factory else None
        return Receiver(self.layout, self.coder), transform

    @staticmethod
    def _rank(results):
        return max((r.identity == 'verified_header', r.coverage,
                    -float(r.pilot_error if r.pilot_error is not None else 1e9))
                   for r in results)

    def _choose(self, found):
        if not found:
            return []
        pair = max(found, key=lambda p: self._rank(found[p]))
        results = found[pair]
        verified = any(r.identity == 'verified_header' for r in results)
        if verified:
            self.selected = pair
            self.candidates = {pair: self.candidates[pair]}
            self.history = np.empty((0, self.channels), np.float32)
        # A damaged header must not prevent a recovered picture from appearing.
        # Without a verified header, continue searching other channel pairs.
        for result in results:
            result.extra['input_channels'] = [c + 1 for c in pair]
            result.extra['channel_lock'] = self.selected is not None
        return results

    def feed(self, block):
        block = np.asarray(block, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != self.channels or not np.isfinite(block).all():
            raise ValueError('Input must contain finite samples for every device channel')
        self.samples += len(block)
        new_pairs = set()
        if self.selected is None and len(block):
            self.history = np.concatenate((self.history, block))[-(4*self.layout.frame+512):]
            # Skip digital silence only: quiet modem audio must remain eligible.
            active = np.flatnonzero(np.max(np.abs(self.history), axis=0) > 1e-12)
            # The transport's stereo channel estimate already corrects swaps.
            for pair in combinations(active.tolist(), 2):
                if pair not in self.candidates:
                    self.candidates[pair] = self._new()
                    self.candidates[pair][0].offset = self.samples - len(self.history)
                    new_pairs.add(pair)
        found = {}
        for pair, (receiver, transform) in self.candidates.items():
            audio = (self.history if pair in new_pairs else block)[:, pair]
            results = receiver.feed(transform.process(audio) if transform else audio)
            if results:
                found[pair] = results
        return self._choose(found)

    def flush(self):
        found = {}
        for pair, (receiver, _) in self.candidates.items():
            results = receiver.flush()
            if results:
                found[pair] = results
        return self._choose(found)
