"""Nested fold: more luma detail in the mono and two-channel packets, with a
receiver that decodes whatever each packet still holds.

Two experimental mappings ride on existing wires and change only what their
luma slots (and the amplitude pattern of the 16 signature slots) carry:

``aspect-mono-500`` + nested fold
    Every luma slot keeps its own coefficient (the *host*) on a staircase and
    most carry one more coefficient (a *guest*) inside the stair.

``stereo-slices`` + nested fold
    Each channel keeps the stock slices slot value as its host (the base
    coefficient plus, on the left, or minus, on the right, a share of a
    detail coefficient) on a staircase, with a guest inside the stair that
    differs between the channels.  So everything the stock profile promises
    still holds: either channel alone is a whole picture, a mono sum is the
    base picture, and both together separate base from detail.  On top, both
    channels together deliver two further sets of guests; and when noise
    rises the guests fade and what is left is the stock slices picture.

One frozen table per layout serves every link.  The receiver measures each
packet's noise on the signature slots and decodes every slot by posterior
mean at that noise (``tools/v7_sk_fold.py``): on a quiet link stairs and
guests are read exactly; as noise grows the guests fade and the hosts slide
to a plain shrunk reading.  Nothing switches and nothing is thresholded.

A picture's overall level is normalised by the sender (a 15-step ladder,
factor 1.5 a step) and signalled as *which* of 15 orthogonal sign patterns
the signature slots carry.  The stock fold's signature is the sixteenth
pattern, so a receiver tells a nested packet from a stock one, per packet,
without a setting, and stock packets decode exactly as before.

Tables: ``nested_tables.npz`` beside this file, built by
``tools/v7_nested_build.py`` from general photographs.  A layout without a
table simply stays on the stock mapping.
"""
import math
import os
import threading
from pathlib import Path

import sys

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools import v7_sk_fold as sk                                      # noqa: E402

# NESTED_FOLD_TABLES points tests and held-out scoring at another build.
TABLES = Path(os.environ.get('NESTED_FOLD_TABLES') or
              Path(__file__).resolve().with_name('nested_tables.npz'))
RATIO, LOWEST, HIGHEST = 1.5, -3, 11
SIGNATURE_SLOTS = 16
SCORE_MIN = .5                 # signature correlation that marks a nested packet
DOUBTFUL = .5                  # equaliser confidence below which a slot's own noise is used
LUMA = 96*80


def hadamard():
    h = np.ones((1, 1))
    while len(h) < SIGNATURE_SLOTS:
        h = np.block([[h, h], [h, -h]])
    return h


HADAMARD = hadamard()


def level_row(index):
    """Sign pattern (over the profile's own signature pattern) for a level."""
    return HADAMARD[int(index)-LOWEST+1]


def read_signature(unit):
    """(score, level, residual rms) from signature slots already divided by
    the profile's pattern and signature amplitude (so a stock packet reads
    all ones)."""
    unit = np.asarray(unit, float)
    correlations = HADAMARD@unit/len(unit)
    row = int(np.argmax(correlations[1:]))+1
    residual = float(np.sqrt(np.mean((unit-HADAMARD[row])**2)))
    return float(correlations[row]), row-1+LOWEST, residual, float(correlations[0])


def loudness(symbols):
    """Noise multiplier of a packet: the sender holds every packet at one
    audio level, so slot noise scales with the slots' own size."""
    return max(.5, 1.2*float(np.sqrt(np.mean(np.square(symbols)))))


class FrozenFold:
    """One layout's frozen nested-fold table and the per-frame code."""

    FIELDS = ('model_index', 'host_position', 'host_mean', 'host_sd', 'noise',
              'signature_noise', 'level', 'step', 'scale', 'alpha', 'compander',
              'density', 'member', 'host_sector', 'law', 'limits', 'guest_position',
              'guest_sd', 'guest_slot', 'guest_sector', 'guest_band')
    # Two-channel tables only: per slot the detail coefficient that rides on
    # the host (left +, right -), its share, and the variances of the base
    # and of the riding detail.
    PAIRED = ('detail_position', 'ride', 'common', 'differ')

    def __init__(self, arrays):
        for name in self.FIELDS:
            setattr(self, name, np.ascontiguousarray(arrays[name]))
        self.paired = all(name in arrays for name in self.PAIRED)
        for name in self.PAIRED:
            setattr(self, name, np.ascontiguousarray(arrays[name]) if self.paired else None)
        self.level = self.level.astype(np.int64)
        self.host_sector = self.host_sector.astype(np.int64)
        self.member = self.member.astype(np.bool_)
        self.inverse = sk.expand_table(self.compander)
        self.density_sum = sk.self_convolved(self.density)
        self.sectors = self.law.shape[0]
        self.legs = self.guest_position.shape[0]
        self.slots = len(self.host_position)
        self._draw = np.random.default_rng(20261006).standard_normal(self.slots)
        mask = np.zeros(LUMA, bool)
        mask[self.host_position] = True
        mask[self.guest_position.ravel()] = True
        if self.paired:
            mask[self.detail_position[self.ride > 0]] = True
        mask.setflags(write=False)
        self.sent_mask = mask

    # ------------------------------------------------------------ both ends
    def _activity(self, hosts):
        total, count = np.zeros(self.sectors), np.zeros(self.sectors)
        picked = hosts[self.member]
        np.add.at(total, self.host_sector[self.member], picked*picked)
        np.add.at(count, self.host_sector[self.member], 1.0)
        return np.maximum(np.sqrt(total/np.maximum(count, 1.0)), .05)

    def _guest_scale(self, activity, leg):
        law = self.law[self.guest_sector[leg], self.guest_band]
        value = np.exp(law[:, 0]+law[:, 1]*np.log(activity[self.guest_sector[leg]]))
        return np.clip(value, self.limits[0], self.limits[1])

    def _stairs(self, hosts):
        """Host values as a slip-free receiver decodes them."""
        return sk.stair_values(np.ascontiguousarray(hosts, float), self.level, self.step)

    # --------------------------------------------------------------- sender
    def _normalised(self, plane, index, leg=0):
        factor = RATIO**index
        flat = np.asarray(plane, float).ravel()
        value = flat[self.host_position]
        if self.paired:
            # The stock slices slot: base plus (left) or minus (right) detail.
            value = value+(1.0 if leg == 0 else -1.0)*self.ride*flat[self.detail_position]
        return (value-self.host_mean)/(self.host_sd*factor), flat, factor

    def encode(self, plane, index, leg=0):
        """Unit-power luma slot values for one channel.  ``plane``: the 96x80
        luma coefficients (any shape of 7,680)."""
        hosts, flat, factor = self._normalised(plane, index, leg)
        scale = self._guest_scale(self._activity(self._stairs(hosts)), leg)
        guests = flat[self.guest_position[leg]]/(self.guest_sd[leg]*factor*scale)
        b = np.zeros(self.slots)
        b[self.guest_slot] = guests
        return sk.encode_frame(hosts, b, np.zeros(self.slots), self.level, self.step,
                               self.scale, self.alpha, 3, .25, self.compander, 1.0)

    def choose_level(self, plane):
        """The ladder level with the least error in a trial decode at the
        noise that packet will meet (nearest-stair decode, fixed draw)."""
        best, chosen = np.inf, 0
        flat = np.asarray(plane, float).ravel()
        for index in range(LOWEST, HIGHEST+1):
            hosts, _, factor = self._normalised(flat, index)
            scale = self._guest_scale(self._activity(self._stairs(hosts)), 0)
            guests = flat[self.guest_position[0]]/(self.guest_sd[0]*factor*scale)
            b = np.zeros(self.slots)
            b[self.guest_slot] = guests
            symbols = sk.encode_frame(hosts, b, np.zeros(self.slots), self.level, self.step,
                                      self.scale, self.alpha, 3, .25, self.compander, 1.0)
            received = symbols+loudness(symbols)*self.noise*self._draw
            got_hosts, got = sk.hard_decode_frame(received, self.level, self.step,
                                                  self.scale, self.alpha, self.inverse)
            got_scale = self._guest_scale(self._activity(got_hosts), 0)
            error = factor*factor*(
                float(np.sum((self.host_sd*(got_hosts-hosts))**2)) +
                float(np.sum((self.guest_sd[0]*(got[self.guest_slot]*got_scale-guests*scale))**2)))
            if error < best:
                best, chosen = error, index
        return chosen

    # ------------------------------------------------------------- receiver
    def decode(self, readings, index):
        """Luma coefficients (flat 7,680) from what arrived.

        ``readings``: {leg: (slot values, slot noise)} with leg 0 and/or 1,
        or {'sum': (...)} for a mono sum of both channels (hosts only).
        Noise is per slot, in units of the slot's design signal.
        """
        factor = RATIO**index
        out = np.zeros(LUMA)
        alpha = self.alpha
        if self.paired:
            return self._decode_paired(readings, factor)
        if 'sum' in readings:
            values, sigma = readings['sum']
            hosts, _ = sk.soft_decode_frame(
                np.ascontiguousarray(values, float), self.level, self.step, self.scale, alpha,
                np.ascontiguousarray(sigma, float), self.density_sum, self.inverse)
            guests = {}
        elif len(readings) == 2:
            (left, sigma_left), (right, sigma_right) = readings[0], readings[1]
            hosts, first, second = sk.soft_decode_pair(
                np.ascontiguousarray(left, float), np.ascontiguousarray(right, float),
                self.level, self.step, self.scale, alpha,
                np.ascontiguousarray(sigma_left, float),
                np.ascontiguousarray(sigma_right, float), self.density, self.inverse)
            guests = {0: first, 1: second}
        else:
            (leg, (values, sigma)), = readings.items()
            hosts, got = sk.soft_decode_frame(
                np.ascontiguousarray(values, float), self.level, self.step, self.scale, alpha,
                np.ascontiguousarray(sigma, float), self.density, self.inverse)
            guests = {leg: got}
        out[self.host_position] = self.host_mean+hosts*self.host_sd*factor
        activity = self._activity(hosts)
        for leg, got in guests.items():
            out[self.guest_position[leg]] = (got[self.guest_slot]*self._guest_scale(activity, leg) *
                                             self.guest_sd[leg]*factor)
        return out


    def _decode_paired(self, readings, factor):
        """Two-channel tables: each channel is decoded on its own stairs,
        then base and detail are separated as the stock slices profile does,
        weighted by what each channel's noise left of them."""
        out = np.zeros(LUMA)
        lam = self.common+self.differ
        quantised = np.where(self.level > 0, self.step**2/12.0, 0.0)
        got = {}
        for key, reading in readings.items():
            values, sigma = reading[0], np.ascontiguousarray(reading[1], float)
            hosts, guests = sk.soft_decode_frame(
                np.ascontiguousarray(values, float), self.level, self.step, self.scale,
                self.alpha, sigma, self.density_sum if key == 'sum' else self.density,
                self.inverse)
            got[key] = (hosts, guests, lam*(sigma*sigma+quantised))
        riding = self.ride > 0
        share = np.where(riding, self.ride, 1.0)
        if 'sum' in got:
            base, detail = got['sum'][0]*self.host_sd, None
        elif len(got) == 2:
            (left, _, noise_left), (right, _, noise_right) = got[0], got[1]
            noise = .5*(noise_left+noise_right)
            already = lam/(lam+noise)
            base = .5*(left+right)*self.host_sd*np.minimum(
                self.common/(self.common+.5*noise)/already, 2.0)
            detail = .5*(left-right)*self.host_sd/share*(
                self.differ/(self.differ+.5*noise)/already)
        else:
            (leg, (hosts, _, _)), = got.items()
            value = hosts*self.host_sd
            base = value*self.common/lam
            detail = (1.0 if leg == 0 else -1.0)*value*self.differ/(lam*share)
        out[self.host_position] = self.host_mean+base*factor
        if detail is not None:
            out[self.detail_position[riding]] = detail[riding]*factor
        for leg, (hosts, guests, _) in got.items():
            if leg == 'sum':
                continue
            out[self.guest_position[leg]] = (
                guests[self.guest_slot]*self._guest_scale(self._activity(hosts), leg) *
                self.guest_sd[leg]*factor)
        return out


_lock = threading.Lock()
_cache = {}


def table(family, layout, path=None):
    """The frozen table for ``family`` ('mono' or 'slices') and ``layout``,
    or None when none was built."""
    path = Path(path or TABLES)
    key = (str(path), family, layout)
    with _lock:
        if key not in _cache:
            fold = None
            if path.is_file():
                with np.load(path) as data:
                    prefix = f'{family}/{layout}/'
                    if prefix+'host_position' in data.files:
                        names = FrozenFold.FIELDS+tuple(
                            name for name in FrozenFold.PAIRED if prefix+name in data.files)
                        fold = FrozenFold({name: data[prefix+name] for name in names})
            _cache[key] = fold
        return _cache[key]


SLOT_NOISE_SHARE = float(os.environ.get('NESTED_SLOT_NOISE_SHARE', .4))


def slot_noise(relative, design, confidence=None, variance=None):
    """Per-slot noise for the soft decoder, in units of each slot's signal.

    Two measurements, and the larger wins slot by slot:

    * the packet's noise read on the signature slots, spread over the table's
      design profile (right for hiss, which is alike everywhere);
    * the equaliser's own figure for that slot, from its confidence or noise
      variance (right for damage that hits some slots and not others: a
      band-limited or aliased link, a dropout, one weak channel).  The
      equaliser overstates clean-link noise, so only SLOT_NOISE_SHARE of it
      is taken.
    """
    sigma = max(relative, .5)*design
    own = None
    if confidence is not None:
        c = np.clip(np.asarray(confidence, float), 1e-3, 1.0)
        own = np.sqrt((1-c)/c)
    if variance is not None:
        own = np.sqrt(np.maximum(np.asarray(variance, float), 0.0))
    if own is not None:
        sigma = np.maximum(sigma, SLOT_NOISE_SHARE*own)
    return sigma


STICKY_PACKETS = 36            # about three seconds of packets


class Recognition:
    """Which mapping a stream is carrying, remembered across packets.

    A packet whose signature slots are readable says what it is (nested, and
    at which level, or stock).  One whose signature is damaged beyond reading
    (a band-limited link, a dropout across those slots) is taken to be what
    the stream was carrying a moment ago, rather than decoded as the wrong
    mapping.
    """

    def __init__(self):
        self.kind, self.level, self.age = None, 0, 0

    def decide(self, score, level, stock_score):
        """(is nested, level) for a packet with these signature readings."""
        if score >= SCORE_MIN and score > stock_score:
            self.kind, self.level, self.age = 'nested', level, 0
        elif stock_score >= SCORE_MIN and stock_score > score:
            self.kind, self.age = 'stock', 0
        else:
            self.age += 1
            if self.age > STICKY_PACKETS:
                self.kind = None
            if self.kind is None:
                # Never seen a readable packet: go by the better reading.
                return score > stock_score and score > .25, level
        return self.kind == 'nested', self.level

    def reset(self):
        self.kind, self.age = None, 0


def packet_noise(residual, design, own=None):
    """The packet-wide (hiss-like) noise relative to the table's design.

    ``residual``: RMS distance of the signature slots from their pattern, in
    slot signal units; ``design``: the table's design noise there; ``own``:
    the equaliser's own noise figure for those slots.  What the equaliser
    already attributes to those particular slots (they sit in a damaged band,
    say) is not spread over the whole packet.
    """
    excess = residual*residual
    if own is not None:
        excess -= float(np.mean((SLOT_NOISE_SHARE*np.asarray(own, float))**2))
    return math.sqrt(max(excess, 0.0))/max(design, 1e-12)


class NestedMonoCodec:
    """A stock mono fold codec that can also send and read the nested fold.

    Everything not overridden is the stock codec's.  ``decode`` reads the
    signature of each packet: nested packets take the soft decoder, stock
    packets the stock decoder.
    """

    def __init__(self, stock, fold, send=False):
        self.stock, self.fold, self.send = stock, fold, bool(send)
        index = fold.model_index
        if not np.array_equal(np.asarray(stock.kept)[index], fold.host_position):
            raise ValueError('nested fold table does not match this layout')
        self.index = index
        self.mu = np.asarray(stock.model.mu)[index]
        self.sd = np.sqrt(np.asarray(stock.model.lam)[index])
        self.signature_index = np.asarray(stock.hosts)[-SIGNATURE_SLOTS:]
        self.signature_sd = np.asarray(stock.sd_host)[-SIGNATURE_SLOTS:]
        from folding import SIGNATURE_STEPS
        self.signature_unit = SIGNATURE_STEPS*stock.D/math.sqrt(stock.power)
        self.last_level = None
        self.last_nested = False
        self.recognition = Recognition()

    def __getattr__(self, name):
        return getattr(self.stock, name)

    def sent_luma_mask(self):
        return self.fold.sent_mask if self.send else self.stock.sent_luma_mask()

    def encode_coefficients(self, values, full=None):
        if full is None:
            full = self.stock.grid.forward(values)
        coefficients = np.array(self.stock.encode_coefficients(values, full), float)
        if not self.send:
            return coefficients
        plane = np.asarray(full)[:LUMA]
        level = self.fold.choose_level(plane)
        coefficients[self.index] = self.mu+self.sd*self.fold.encode(plane, level, 0)
        coefficients[self.signature_index] = (
            np.asarray(self.stock.model.mu)[self.signature_index] +
            self.signature_sd*self.signature_unit*self.stock.pattern*level_row(level))
        return coefficients

    def decode(self, coeffs, xhat, conf, fallback=True, metadata_confirmed=False):
        confidence = np.clip(np.asarray(conf, float), 1e-3, 1.0)
        seen = np.asarray(xhat, float)/confidence
        unit = (seen[self.signature_index]/self.signature_sd /
                self.signature_unit*self.stock.pattern)
        score, level, residual, stock_score = read_signature(unit)
        self.last_nested, level = self.recognition.decide(score, level, stock_score)
        if not self.last_nested:
            return self.stock.decode(coeffs, xhat, conf, fallback=fallback,
                                     metadata_confirmed=metadata_confirmed)
        fold = self.fold
        doubt = confidence[self.signature_index]
        relative = packet_noise(residual*self.signature_unit,
                                float(np.sqrt(np.mean(fold.signature_noise**2))),
                                np.sqrt((1-doubt)/doubt))
        sigma = slot_noise(relative, fold.noise, confidence[self.index])
        full = self.stock.plain(coeffs)
        full[:LUMA] = fold.decode({0: (seen[self.index]/self.sd, sigma)}, level)
        self.stock.last_score = self.last_score = score
        self.stock.last_noise = self.last_noise = relative
        self.stock.last_unfolded_slots = self.last_unfolded_slots = int(np.sum(fold.level > 0))
        self.last_level = level
        return full


def enable_mono(wire, send=False, path=None):
    """Make an ``AspectMonoWire`` read nested-fold packets (and send them
    with ``send``).  Layouts without a table are left on the stock codec."""
    real = wire._codec
    wrapped = {}

    def codec(model):
        stock = real(model)
        fold = table('mono', getattr(stock, 'layout', None), path)
        if fold is None:
            return stock
        if id(stock) not in wrapped:
            wrapped[id(stock)] = NestedMonoCodec(stock, fold, send)
        return wrapped[id(stock)]

    def forget():
        for nested in wrapped.values():
            nested.recognition.reset()

    wire._codec = codec
    wire.nested_fold = 'send' if send else 'receive'
    wire.reset_nested = forget          # a new stream: forget what the last one carried
    if send:
        # Nested packets go out auto-levelled: the fold leaves some pictures'
        # bodies well under the ceiling, and that headroom is signal-to-noise.
        from animation_modem import v7
        encode_packet = wire.encode_packet

        def levelled(*args, **kwargs):
            with v7.body_auto_level():
                return encode_packet(*args, **kwargs)

        wire.encode_packet = levelled
    return wire


def _slice_wire_class():
    import slice_wire as stock

    class NestedHalf(stock.Half):
        __slots__ = ('signature',)

    class NestedSliceWire(stock.SliceWire):
        """Stereo slices whose channels are nested-fold mono pictures with
        shared hosts and different guests.  Reads stock slices unchanged."""

        def __init__(self, layout='auto', send=False, path=None):
            super().__init__(layout)
            self.send, self._path = bool(send), path
            self._level = (None, None)
            self.nested_fold = 'send' if send else 'receive'
            self.last_nested = False
            self.recognition = Recognition()

        def reset(self):
            super().reset()
            self.recognition.reset()

        def fold(self, layout):
            fold = table('slices', layout, self._path)
            if fold is not None and not np.array_equal(
                    fold.host_position, self.slots(layout).base[0]):
                raise ValueError('nested fold table does not match this layout')
            return fold

        def luma_sent_mask(self, layout):
            fold = self.fold(layout) if self.send else None
            return super().luma_sent_mask(layout) if fold is None else fold.sent_mask

        # ------------------------------------------------------------ sender
        def encode_packet(self, *args, **kwargs):
            # Nested packets go out auto-levelled (see enable_mono).
            with stock.v7.body_auto_level(self.send):
                return super().encode_packet(*args, **kwargs)

        def channel_coefficients(self, model, spectra, kind):
            out = super().channel_coefficients(model, spectra, kind)
            layout = self._layout_of[id(model)]
            fold = self.fold(layout) if self.send else None
            if fold is None:
                return out
            slots = self.slots(layout)
            plane = spectra[0]
            # Both channels of a frame share one level: choose it once.
            if self._level[0] is not plane:
                self._level = (plane, fold.choose_level(plane))
            level = self._level[1]
            part = slots.parts[0]
            symbols = fold.encode(plane, level, stock.SIDES.index(kind))
            out[part] = slots.model_mu[part]+np.sqrt(slots.model_lam[part])*symbols
            hosts = slots.fold_hosts[slots.guests:]
            out[hosts] = slots.model_mu[hosts]+slots.sd_host[slots.guests:]*(
                stock.SIGNATURE_STEPS*stock.FOLD_STEP*stock.SIGNATURE_PATTERN *
                level_row(level))/np.sqrt(stock.FOLD_POWER)
            return out

        # ---------------------------------------------------------- receiver
        def half(self, model, result, channel=None):
            plain = super().half(model, result, channel)
            slots = self.slots(plain.layout)
            equalized = result.diag.get('mono_fold_eq')
            if equalized is not None:
                trust = np.clip(np.asarray(equalized[1], float), 1e-3, 1.0)
                seen = np.asarray(equalized[0], float)/trust
            else:
                seen = np.asarray(result.coeffs, float)-slots.model_mu
            hosts = slots.fold_hosts[slots.guests:]
            unit = (seen[hosts]/slots.sd_host[slots.guests:]*np.sqrt(stock.FOLD_POWER) /
                    (stock.SIGNATURE_STEPS*stock.FOLD_STEP)*stock.SIGNATURE_PATTERN)
            half = NestedHalf(plain.kind, plain.score, plain.layout, plain.good, plain.noise,
                              plain.damaged, plain.steady, plain.raw, plain.ring)
            own = None
            if equalized is not None:
                own = np.sqrt((1-trust[hosts])/trust[hosts])
            half.signature = read_signature(unit)+(own,)
            return half

        @staticmethod
        def nested_crosstalk(halves):
            """The share of each channel heard in the other, from the
            markers, as the stock wire reads it but without its upper bound:
            a stair cannot be read through crosstalk the stock profile would
            merely blur, so every share the markers can measure is undone."""
            sizes = [abs(half.steady) for half in halves]
            if abs(sizes[0]-sizes[1]) > 3*stock.CROSSTALK_AGREE:
                return 0.0
            ratio = min(sum(sizes)/2, 1.0)
            leak = (1-ratio)/(1+ratio)
            return float(leak) if leak >= stock.CROSSTALK_MIN/2 else 0.0

        def values(self, halves):
            chosen = self.choose(halves)
            fold = self.fold(chosen[0].layout) if chosen else None
            signed = [getattr(half, 'signature', None) for half in chosen]
            level = 0
            if fold is None or not signed or any(s is None for s in signed):
                self.last_nested = False
            else:
                # The channel whose signature reads best speaks for the packet.
                best = max(signed, key=lambda s: max(s[0], s[3]))
                self.last_nested, level = self.recognition.decide(best[0], best[1], best[3])
            if not self.last_nested:
                return super().values(halves)
            slots = self.slots(chosen[0].layout)
            lam = slots.lam[0]
            unit_scale = stock.SIGNATURE_STEPS*stock.FOLD_STEP/np.sqrt(stock.FOLD_POWER)
            design = max(float(np.sqrt(np.mean(fold.signature_noise**2))), 1e-12)
            kinds = [half.kind for half in chosen]
            leak = self.nested_crosstalk(chosen) if sorted(kinds) == sorted(stock.SIDES) else 0.0
            seen = {half.kind: half.raw[0][0] for half in chosen}
            if leak:
                a, b = seen['left'], seen['right']
                seen = {'left': (a-leak*b)/(1-leak), 'right': (b-leak*a)/(1-leak)}
            readings = {}
            for half, (_, _, residual, _, own) in zip(chosen, signed):
                sigma = slot_noise(packet_noise(residual*unit_scale, design, own), fold.noise,
                                   variance=half.raw[0][1]/np.maximum(lam, 1e-30))
                key = 'sum' if half.kind == 'sum' else stock.SIDES.index(half.kind)
                readings[key] = (seen[half.kind]/np.sqrt(lam), sigma)
            if 'sum' in readings:
                readings = {'sum': readings['sum']}
            luma = fold.decode(readings, level)
            out = []
            for plane, ((base, _), (rows, cols)) in enumerate(
                    zip(self.estimate(chosen), stock.v7.V7_GRIDS)):
                if plane == 0:
                    spectrum = luma
                else:
                    spectrum = np.zeros(rows*cols)
                    spectrum[slots.base[plane]] = base
                out.append((stock._dct_matrix(rows).T @ spectrum.reshape(rows, cols) @
                            stock._dct_matrix(cols)).ravel())
            return np.concatenate(out)

    return NestedSliceWire


def slice_wire(layout='auto', send=False, path=None):
    """A stereo-slices wire that reads nested-fold packets (and sends them
    with ``send``)."""
    return _slice_wire_class()(layout, send=send, path=path)
