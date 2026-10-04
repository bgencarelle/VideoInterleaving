"""Temporal fusion of decoded V7 packets: a held picture gets sharper.

Every packet is decoded on its own: the equaliser returns, per coefficient,
an LMMSE estimate and a confidence against the model's variance. On a noisy
link that estimate is shrunk towards the model mean (the picture softens) and
the fold's guests are dropped once their symbol noise passes the table's
limit. When the sender holds a picture (a still, a paused file, a slow
scene) successive packets are repeated observations of the same
coefficients, and that noise averages out.

This is a per-coefficient Kalman filter on the equaliser output, before the
fold is undone:

- an observation is the unshrunk value ``xhat/conf`` with noise variance
  ``lam*(1-conf)/conf`` (what the LMMSE confidence means);
- the state carries a value and a variance per coefficient;
- how much the picture moved since the last packet is measured on the
  packet-to-packet difference of every coefficient both packets carry: the
  power beyond what the two noise variances explain, as a fraction of the
  model variance, averaged in eight groups along the model's rank (a small
  shift moves fine detail far more than coarse, so one figure for the whole
  picture would miss it). That is added to the state variance before the
  update, so a moving picture forgets the state at once
  (the output is then the packet as decoded, as before) and a held one keeps
  averaging;
- the equaliser's confidence is pessimistic (measured: the real error is
  0.2 to 0.7 of what it implies), which would hide slow motion under noise
  that is not there. The noise is therefore calibrated on the second
  difference of three packets, which a steady drift cancels and noise does
  not; the calibration only ever lowers the claimed noise;
- the fused value is shrunk again with its own, higher, confidence, which is
  what the fold decoder and the display gate read.

Nothing is fused across a change of model, profile or direction, across a
packet that was not accepted, or across a gap. A rotating tail slice is fused on the packets that carry
it and otherwise left to the decoder's tail memory.
"""
import numpy as np

# Confidence below which a coefficient of this packet is not an observation.
MIN_CONFIDENCE = 1e-3
# Packets of averaging a perfectly still picture is limited to. It bounds how
# long a change too small to measure can lag (about 2 s at 12.2 pictures/s).
MAX_PACKETS = 24.0
# Measured motion (difference power / model variance) above which the state
# is dropped outright instead of being down-weighted.
RESET_MOTION = .5
# The proportional motion term is floored here.
MIN_MOTION = 1e-5
# Bounds and smoothing of the noise calibration (never above 1: the
# confidence was not seen to be optimistic).
MIN_NOISE_SCALE = .05
NOISE_SCALE_MEMORY = .5
# Groups of coefficients, along the model's rank, that motion is measured in.
MOTION_BANDS = 8
# Standard errors added to each group's measured motion, and the share of
# the previous packet's motion that is still assumed (motion persists).
MOTION_MARGIN = 1.0
MOTION_HOLD = .5


def _observation(model, xhat, conf):
    lam = np.asarray(model.lam, float)
    conf = np.clip(np.asarray(conf, float), 0.0, 1.0-1e-9)
    seen = conf > MIN_CONFIDENCE
    safe = np.where(seen, conf, 1.0)
    value = np.where(seen, np.asarray(xhat, float)/safe, 0.0)
    variance = np.where(seen, lam*(1.0-safe)/safe, np.inf)
    return value, variance, seen


def band_motion(difference, noise, lam, order, bands=MOTION_BANDS):
    """Motion per coefficient, as a fraction of its model variance.

    ``order`` ranks the coefficients (most important first). They are cut
    into ``bands`` groups of equal count along that order, and each group's
    motion is the mean of ``(difference**2 - noise)/lam``, floored. A group
    spans a narrow range of picture frequency, so a small shift, which moves
    fine detail far more than coarse, is measured where it shows.

    The mean is raised by MOTION_MARGIN standard errors of a noise-only
    group: on a noisy link a real motion can measure as none by chance, and
    averaging a moving picture costs more than not averaging a still one.
    """
    scaled = noise/lam
    relative = difference*difference/lam-scaled
    position = np.argsort(np.argsort(order, kind='stable'), kind='stable')
    group = np.minimum(position*bands//max(len(order), 1), bands-1)
    count = np.maximum(np.bincount(group, minlength=bands), 1)
    mean = np.bincount(group, weights=relative, minlength=bands)/count
    # A noise-only difference squared has variance 2 noise^2.
    error = np.sqrt(2.0*np.bincount(group, weights=scaled*scaled,
                                    minlength=bands))/count
    motion = np.maximum(mean+MOTION_MARGIN*error, MIN_MOTION)
    return motion[group], motion


class TemporalFusion:
    """Kalman fusion of successive packets' equaliser outputs."""

    def __init__(self, max_packets=MAX_PACKETS):
        self.max_packets = float(max_packets)
        self.reset()

    def reset(self):
        self._key = None
        self._value = self._variance = None
        self._history = []
        self._rank = None
        self._bands = None
        self._noise_scale = 1.0
        self.last_motion = None
        self.last_gain = 1.0

    def update(self, model, xhat, conf, key=None):
        """Fuse one accepted packet. Returns ``(xhat, conf, known)``: the
        fused estimate and confidence in the equaliser's convention, and the
        coefficients they describe (the others keep the packet's values)."""
        lam = np.asarray(model.lam, float)
        value, variance, seen = _observation(model, xhat, conf)
        key = (key, id(model), len(lam))
        if self._key != key or self._value is None:
            self.reset()
            self._key = key
            self._value, self._variance = value, variance
            self._history = [(value, variance, seen)]
            return np.asarray(xhat, float), np.asarray(conf, float), seen

        last_value, last_variance, last_seen = self._history[-1]
        if len(self._history) >= 2:
            # Noise calibration on the second difference (variance 6 r for
            # equal noise; a steady drift cancels).
            first_value, first_variance, first_seen = self._history[-2]
            triple = seen & last_seen & first_seen
            if np.count_nonzero(triple) >= 64:
                second = (value-2.0*last_value+first_value)[triple]
                claimed = (variance+4.0*last_variance+first_variance)[triple]
                scale = float(np.sum(second*second/lam[triple]) /
                              max(np.sum(claimed/lam[triple]), 1e-300))
                scale = min(max(scale, MIN_NOISE_SCALE), 1.0)
                self._noise_scale += (1.0-NOISE_SCALE_MEMORY)*(
                    scale-self._noise_scale)
        scale = self._noise_scale
        self._history = (self._history+[(value, variance, seen)])[-2:]

        both = seen & last_seen
        rank = getattr(self, '_rank', None)
        if rank is None or len(rank) != len(lam):
            rank = np.empty(len(lam), dtype=np.int64)
            rank[np.asarray(model.order)] = np.arange(len(lam))
            self._rank = rank
        relative = np.full(len(lam), np.nan)
        if np.count_nonzero(both) < 64:
            head_motion = RESET_MOTION
        else:
            relative[both], bands = band_motion(
                (value-last_value)[both],
                scale*(variance+last_variance)[both], lam[both], rank[both])
            held = getattr(self, '_bands', None)
            if held is not None and len(held) == len(bands):
                raised = np.maximum(bands, MOTION_HOLD*held)
                relative[both] *= (raised/bands)[np.minimum(
                    np.argsort(np.argsort(rank[both], kind='stable'),
                               kind='stable')*len(bands) //
                    max(int(np.count_nonzero(both)), 1), len(bands)-1)]
                bands = raised
            self._bands = bands
            head_motion = float(bands[0])
            # Coefficients this pair of packets does not both carry (the
            # rotating tail) move like the finest group measured.
            relative[~both] = float(bands[-1])
        self.last_motion = head_motion
        if head_motion >= RESET_MOTION:
            self._value, self._variance = value, variance
            self.last_gain = 1.0
            return np.asarray(xhat, float), np.asarray(conf, float), seen

        # Predict (the picture moved), bound the memory, then update.
        noise = scale*variance
        state_variance = self._variance+relative*lam
        if self.max_packets > 0:
            state_variance = np.where(
                seen, np.maximum(state_variance, noise/self.max_packets),
                state_variance)
        known = np.isfinite(state_variance)
        precision_state = np.where(
            known, 1.0/np.maximum(state_variance, 1e-300), 0.0)
        precision_new = np.where(seen, 1.0/np.maximum(noise, 1e-300), 0.0)
        total = precision_state+precision_new
        any_seen = total > 0
        safe_total = np.where(any_seen, total, 1.0)
        fused = np.where(any_seen, (precision_state*self._value +
                                    precision_new*value)/safe_total, 0.0)
        fused_variance = np.where(any_seen, 1.0/safe_total, np.inf)
        self._value, self._variance = fused, fused_variance
        # Report confidence on the equaliser's own (uncalibrated) scale, so
        # a packet that gained nothing reads exactly as it was decoded and
        # the fold's confidence limits keep their meaning. Coefficients this
        # packet does not carry stay with the decoder's own tail memory.
        fused_conf = np.where(seen, lam/(lam+np.where(
            seen, fused_variance/scale, 1.0)), 0.0)
        self.last_gain = float(np.median(
            (total/np.maximum(precision_new, 1e-300))[seen])) \
            if seen.any() else 1.0
        return np.where(seen, fused, 0.0)*fused_conf, fused_conf, seen


def fused_packet(fusion, model, coeffs, xhat, conf, key=None):
    """``(coeffs, xhat, conf)`` for the fold decoder, fused over time.

    ``coeffs`` is the decoder's gated result for the packet; the fused
    coefficients replace the ones this packet carries, through the same
    confidence gate.
    """
    from animation_modem import v7
    fused_xhat, fused_conf, seen = fusion.update(model, xhat, conf, key)
    floor = v7._gate_floor(model, False)
    gate = np.clip((fused_conf-floor)/(.85-floor), 0, 1)
    fused_coeffs = np.where(seen, np.asarray(model.mu, float)+fused_xhat*gate,
                            np.asarray(coeffs, float))
    return fused_coeffs, fused_xhat, fused_conf
