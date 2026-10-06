"""Fixed-layout analog fold innovations; no new grids, ranks or wire fields.

Established operations: ridge regression, sparse spectral prediction, and
block Karhunen–Loève rotations. All predictors are frozen on training data.
"""
import copy
import hashlib
import numpy as np

from tools.v7_analog_frame import ProductionFrameWire

METHODS = ('residual-scale', 'local-lift', 'noise-ridge', 'ridge-klt')


class FoldInnovationWire(ProductionFrameWire):
    def __init__(self, profile, layout, method, training, repair='ordinary'):
        if method not in METHODS:
            raise ValueError(method)
        super().__init__(profile, layout, repair)
        self.method = method
        old = self.codec
        self.codec = copy.copy(old)
        codec = self.codec
        usable = codec.M-codec.signature
        self.guests = np.asarray(codec.guests[:usable], int)
        # Only ordinary, current-packet luma coefficients can predict folded
        # detail. Never depend on other guests, quantized hosts or tail memory.
        sent = np.unique(np.concatenate(self.model.rank_tables))
        sent = sent[sent >= 0]
        candidates = np.setdiff1d(sent, codec.hosts)
        positions = codec.kept[candidates]
        candidates = candidates[positions < codec.grid.off[1]]
        positions = codec.kept[candidates]
        cols = codec.grid.grids[0][1]
        u, v = positions//cols, positions % cols
        self.anchors = codec.kept[candidates[np.argsort(u*u+v*v, kind='stable')[:32]]]
        coefficients = np.stack([codec.grid.forward(ProductionFrameWire.values(self, rgb))
                                 for rgb in training])
        if len(coefficients) < 4:
            raise ValueError('need at least four training pictures')
        x, y = coefficients[:, self.anchors], coefficients[:, self.guests]
        self.anchor_mean = x.mean(0)
        self.anchor_sd = np.maximum(x.std(0, ddof=1), .01)
        self.guest_mean = y.mean(0)
        x = (x-self.anchor_mean)/self.anchor_sd
        y = y-self.guest_mean
        # Noise-amplification penalty in standardized anchor coordinates.
        # This is a declared design regularizer, not measured tape noise.
        ridge = .15
        self.weights = np.zeros((len(self.guests), len(self.anchors)))
        anchor_uv = np.stack((self.anchors//cols, self.anchors % cols), -1)
        guest_uv = np.stack((self.guests//cols, self.guests % cols), -1)
        for i in range(len(self.guests)):
            if method == 'residual-scale':
                continue
            selection = (np.argsort(np.sum((anchor_uv-guest_uv[i])**2, axis=1))[:4]
                         if method == 'local-lift' else np.arange(len(self.anchors)))
            a = x[:, selection]
            weight = np.linalg.solve(a.T@a/len(x)+ridge*np.eye(len(selection)),
                                     a.T@y[:, i]/len(x))
            # Bound propagation of normalized anchor error into each guest.
            original_sd = max(float(old.sd_guest[i]), .01)
            weight *= min(1., original_sd/max(np.linalg.norm(weight), 1e-12))
            self.weights[i, selection] = weight
        residual = y-x@self.weights.T
        self.rotations = []
        for first in range(0, usable, 8):
            last = min(first+8, usable)
            if method == 'ridge-klt':
                block = residual[:, first:last]
                cov = block.T@block/len(block)
                _, vectors = np.linalg.eigh(cov)
                rotation = vectors[:, ::-1]
                # Deterministic signs for frozen model identity.
                signs = np.sign(rotation[np.argmax(abs(rotation), axis=0), np.arange(last-first)])
                rotation *= np.where(signs == 0, 1., signs)
            else:
                rotation = np.eye(last-first)
            self.rotations.append((first, last, rotation))
            residual[:, first:last] = residual[:, first:last]@rotation
        codec.sd_guest = old.sd_guest.copy()
        codec.sd_guest[:usable] = np.maximum(np.sqrt(np.mean(residual*residual, axis=0)),
                                            old.sd_guest[:usable]*.05)
        digest = hashlib.sha256()
        for array in (self.anchors, self.guests, self.anchor_mean, self.anchor_sd,
                      self.guest_mean, self.weights, *[r[2] for r in self.rotations]):
            digest.update(np.asarray(array).tobytes())
        self.innovation_digest = digest.hexdigest()
        # Include the predictor's identity in the pre-agreed fold table identity,
        # using the existing signature slots, without adding signaling capacity.
        original_table = codec.table
        codec.table = lambda: dict(original_table(), innovation_method=method,
                                   innovation_digest=self.innovation_digest)
        codec._set_identity()
        if profile == 'aspect-fold-500':
            self.wire._codecs[id(self.model)] = codec
        else:
            self.wire._codec_by_model_id[id(self.model)] = codec
            for key, value in list(self.wire._codecs.items()):
                if value is old:
                    self.wire._codecs[key] = codec
        original_energy = np.mean(y*y)
        self.training_residual_energy_ratio = float(np.mean(residual*residual)/max(original_energy, 1e-12))

    def prediction(self, coefficients):
        x = (coefficients[self.anchors]-self.anchor_mean)/self.anchor_sd
        return self.guest_mean+self.weights@x

    def lift(self, coefficients, inverse=False):
        output = coefficients.copy()
        if inverse:
            for first, last, rotation in self.rotations:
                output[self.guests[first:last]] = coefficients[self.guests[first:last]]@rotation.T
            output[self.guests] += self.prediction(coefficients)
        else:
            residual = coefficients[self.guests]-self.prediction(coefficients)
            for first, last, rotation in self.rotations:
                output[self.guests[first:last]] = residual[first:last]@rotation
        return output

    def values(self, rgb):
        original = ProductionFrameWire.values(self, rgb)
        return self.codec.grid.inverse(self.lift(self.codec.grid.forward(original)))

    def decode(self, audio, rate=96000):
        import time
        started = time.perf_counter()
        packets, info, elapsed = super().decode(audio, rate)
        pictures = [(r, None if values is None else self.codec.grid.inverse(
            self.lift(self.codec.grid.forward(values), inverse=True))) for r, values in packets]
        return pictures, info, time.perf_counter()-started

    def record(self):
        record = super().record()
        record.update(method=self.method, innovation_digest=self.innovation_digest,
                      grids=self.codec.grid.grids, component_pool=(1920, 480, 480),
                      anchors=self.anchors.tolist(), innovation_guests=self.guests.tolist(),
                      anchor_mean=self.anchor_mean.tolist(), anchor_sd=self.anchor_sd.tolist(),
                      guest_mean=self.guest_mean.tolist(), weights=self.weights.tolist(),
                      rotations=[{'first': f, 'last': l, 'matrix': r.tolist()} for f,l,r in self.rotations],
                      training_residual_energy_ratio=self.training_residual_energy_ratio,
                      ridge_penalty=.15, unchanged_rank_tables=True,
                      compression='conditional analog residual energy, not slot-count reduction')
        return record
