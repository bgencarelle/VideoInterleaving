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

Stairs are sent with subtractive dither (``DITHER``): every folded slot's
staircase is shifted by an offset both ends know, one of ``DITHER_PHASES``
offset sets chosen by the packet counter.  A plain staircase makes the same
rounding error in every packet (a dead zone, contours on gradients, static
grain on a held picture); a shifted one makes an error of the same size that
differs from packet to packet, so it averages out over a held picture.  A
dithered packet carries its level pattern negated, so a receiver knows per
packet which staircase to read; the stereo channels use opposite offsets, so
a mono sum of them is read as before.

Tables: ``nested_tables.npz`` beside this file, built by
``tools/v7_nested_build.py`` from general photographs.  A layout without a
table simply stays on the stock mapping.
"""
import math
import os
import threading
import types
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
# Subtractive dither of the stairs; NESTED_FOLD_DITHER=0 sends plain stairs.
DITHER = os.environ.get('NESTED_FOLD_DITHER', '1') != '0'
# The packet counter modulo this picks the offset set.  Seven is the cycle
# both ends already share (the tail slice), forwards and in reverse.
DITHER_PHASES = 7


def hadamard():
    h = np.ones((1, 1))
    while len(h) < SIGNATURE_SLOTS:
        h = np.block([[h, h], [h, -h]])
    return h


HADAMARD = np.ascontiguousarray(hadamard(), np.float64)


def level_row(index, dithered=False):
    """Sign pattern (over the profile's own signature pattern) for a level;
    negated when the packet's stairs are dithered."""
    return HADAMARD[int(index)-LOWEST+1]*(-1.0 if dithered else 1.0)


def _f8(values):
    """A contiguous float64 array: what every kernel takes."""
    return np.ascontiguousarray(values, np.float64)


_NONE = np.zeros(0)


def read_signature(unit):
    """(score, level, residual rms, stock score, dithered) from signature
    slots already divided by the profile's pattern and signature amplitude
    (so a stock packet reads all ones)."""
    score, row, residual, stock, dithered = sk.read_signature(_f8(unit), HADAMARD)
    return float(score), int(row)-1+LOWEST, float(residual), float(stock), bool(dithered)


def loudness(symbols):
    """Noise multiplier of a packet: the sender holds every packet at one
    audio level, so slot noise scales with the slots' own size."""
    return float(sk.loudness(_f8(symbols)))


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
        # Table loading (once): every array is stored in the type and layout
        # the kernels take, so nothing is converted per packet.
        integers = ('model_index', 'host_position', 'level', 'host_sector', 'guest_position',
                    'guest_slot', 'guest_sector', 'guest_band', 'detail_position')
        for name in self.FIELDS:
            kind = np.bool_ if name == 'member' else np.int64 if name in integers else np.float64
            setattr(self, name, np.ascontiguousarray(arrays[name], kind))
        self.paired = all(name in arrays for name in self.PAIRED)
        self.slots = len(self.host_position)
        for name in self.PAIRED:
            kind = np.int64 if name in integers else np.float64
            setattr(self, name, np.ascontiguousarray(arrays[name], kind) if self.paired
                    else np.zeros(self.slots, kind))
        self.inverse = sk.expand_table(self.compander)
        self.density_sum = sk.self_convolved(self.density)
        self.sectors = self.law.shape[0]
        self.legs = self.guest_position.shape[0]
        self._draw = np.random.default_rng(20261006).standard_normal(self.slots)
        # Stair offsets, within half a step, one set per dither phase.
        self._dither = np.ascontiguousarray(np.random.default_rng(20261007).uniform(
            -.5, .5, (DITHER_PHASES, self.slots))*self.step*(self.level > 0))
        mask = np.zeros(LUMA, bool)
        mask[self.host_position] = True
        mask[self.guest_position.ravel()] = True
        if self.paired:
            mask[self.detail_position[self.ride > 0]] = True
        mask.setflags(write=False)
        self.sent_mask = mask
        rows, cols = sk.dct_matrix(96), sk.dct_matrix(LUMA//96)
        self._transforms = (rows, np.ascontiguousarray(rows.T),
                            cols, np.ascontiguousarray(cols.T))

    # ------------------------------------------------------------ both ends
    def _activity(self, hosts):
        return sk.sector_activity(_f8(hosts), self.member, self.host_sector, self.sectors)

    def _guest_scale(self, activity, leg):
        return sk.guest_scales(activity, self.law, self.guest_sector[leg], self.guest_band,
                               self.limits[0], self.limits[1])

    def offset(self, phase, leg=0):
        """Stair offsets of a dithered packet (None: plain stairs).  The
        second channel's are the first's negated."""
        if phase is None:
            return None
        return sk.dither_offsets(self._dither[int(phase) % DITHER_PHASES],
                                 -1.0 if leg == 1 else 1.0)

    def _stairs(self, hosts, offset=None):
        """Host values as a slip-free receiver decodes them."""
        if offset is not None:
            return sk.stair_values_dithered(_f8(hosts), self.level, self.step, offset)
        return sk.stair_values(_f8(hosts), self.level, self.step)

    def _soft(self, values, sigma, density, offset=None):
        if offset is not None:
            return sk.soft_decode_frame_dithered(
                _f8(values), self.level, self.step, self.scale, self.alpha, _f8(sigma),
                density, self.inverse, offset)
        return sk.soft_decode_frame(_f8(values), self.level, self.step, self.scale,
                                    self.alpha, _f8(sigma), density, self.inverse)

    def room(self, level, sigma, count=1, joined=True):
        """How far each luma coefficient (flat 7,680) may lie from its
        decoded value: infinite where nothing was sent, half a stair for a
        folded host (less once ``count`` dithered packets are averaged),
        the noise for a plain host, nothing for a guest.  None when the
        bounds are not known per coefficient (one channel, or a mono sum, of
        a two-channel table: only base plus or minus detail is bounded)."""
        if self.paired and not joined:
            return None
        return sk.coefficient_room(
            LUMA, self.host_position, self.level, self.step, self.host_sd, self.scale,
            _f8(sigma), RATIO**level, SMOOTH_ROOM, int(count), self.guest_position,
            self.ride, self.detail_position, self.paired)

    def clean(self, luma, room, passes=None):
        """The cleanest picture within ``room`` of the decoded ``luma``."""
        passes = SMOOTH_PASSES if passes is None else int(passes)
        if passes <= 0 or room is None:
            return luma
        return sk.smooth_within_room_fast(_f8(luma), room, 96, *self._transforms, passes,
                                     SMOOTH_EDGE, SMOOTH_RATE)

    # --------------------------------------------------------------- sender
    def _normalised(self, plane, index, leg=0):
        factor = RATIO**index
        flat = _f8(plane).ravel()
        # Two-channel tables: the stock slices slot, base plus (left) or
        # minus (right) detail.
        sign = (1.0 if leg == 0 else -1.0) if self.paired else 0.0
        hosts = sk.normalised_hosts(flat, self.host_position, self.host_mean, self.host_sd,
                                    factor, self.ride, self.detail_position, sign)
        return hosts, flat, factor

    def encode(self, plane, index, leg=0, phase=None):
        """Unit-power luma slot values for one channel.  ``plane``: the 96x80
        luma coefficients (any shape of 7,680).  ``phase``: the packet's
        dither phase, or None for plain stairs."""
        hosts, flat, factor = self._normalised(plane, index, leg)
        offset = self.offset(phase, leg)
        scale = self._guest_scale(self._activity(self._stairs(hosts, offset)), leg)
        guests = sk.normalised_guests(flat, self.guest_position[leg], self.guest_sd[leg],
                                      self.guest_slot, scale, factor, self.slots)
        if offset is not None:
            return sk.encode_frame_dithered(hosts, guests, self.level, self.step,
                                            self.scale, self.alpha, self.compander, offset)
        return sk.encode_frame(hosts, guests, _NOTHING(self.slots), self.level, self.step,
                               self.scale, self.alpha, 3, .25, self.compander, 1.0)

    def choose_level(self, plane):
        """The ladder level with the least error in a trial decode at the
        noise that packet will meet (nearest-stair decode, fixed draw)."""
        return int(sk.choose_level(
            _f8(plane).ravel(), LOWEST, HIGHEST, RATIO, self.host_position, self.host_mean,
            self.host_sd, self.level, self.step, self.scale, self.alpha, self.compander,
            self.inverse, self.member, self.host_sector, self.sectors, self.law,
            self.guest_sector[0], self.guest_band, self.limits[0], self.limits[1],
            self.guest_position[0], self.guest_sd[0], self.guest_slot, self.noise, self._draw,
            self.ride, self.detail_position, 1.0 if self.paired else 0.0))

    # ------------------------------------------------------------- receiver
    def _place_guests(self, out, leg, hosts, guests, factor):
        sk.place_guests(out, self.guest_position[leg], guests, self.guest_slot,
                        self._guest_scale(self._activity(hosts), leg),
                        self.guest_sd[leg], factor)

    def soften(self, readings, phase=None):
        """Two-channel tables: every channel's hosts and guests off its own
        stairs, {key: (hosts, guests, noise)}.  This is the costly part of
        ``decode``; pass it as ``soft`` to decode the same readings again
        (shown and ``averaged``) without repeating it."""
        got = {}
        for key, reading in readings.items():
            sigma = _f8(reading[1])
            if key == 'sum':
                hosts, guests = self._soft(reading[0], sigma, self.density_sum)
            else:
                hosts, guests = self._soft(reading[0], sigma, self.density,
                                           self.offset(phase, key))
            got[key] = (hosts, guests, sigma)
        return got

    def decode(self, readings, index, phase=None, averaged=False, soft=None):
        """Luma coefficients (flat 7,680) from what arrived.  ``phase``: the
        dither phase of a dithered packet, or None for plain stairs.
        ``averaged``: the result is for averaging over the dither cycle
        (``Held``), where stair error cancels, so it is not also shrunk for
        that error as a picture shown on its own is.

        ``readings``: {leg: (slot values, slot noise)} with leg 0 and/or 1,
        or {'sum': (...)} for a mono sum of both channels (hosts only).
        Noise is per slot, in units of the slot's design signal.
        """
        factor = RATIO**index
        if self.paired:
            return self._decode_paired(
                self.soften(readings, phase) if soft is None else soft, factor, averaged)
        out = np.zeros(LUMA)
        if 'sum' in readings:
            # Opposite offsets on the two channels cancel in their sum.
            values, sigma = readings['sum']
            hosts, _ = self._soft(values, sigma, self.density_sum)
            guests = {}
        elif len(readings) == 2 and phase is not None:
            raise ValueError('shared-host tables are not sent dithered')
        elif len(readings) == 2:
            (left, sigma_left), (right, sigma_right) = readings[0], readings[1]
            hosts, first, second = sk.soft_decode_pair(
                _f8(left), _f8(right), self.level, self.step, self.scale, self.alpha,
                _f8(sigma_left), _f8(sigma_right), self.density, self.inverse)
            guests = {0: first, 1: second}
        else:
            (leg, (values, sigma)), = readings.items()
            hosts, got = self._soft(values, sigma, self.density, self.offset(phase, leg))
            guests = {leg: got}
        sk.place_hosts(out, self.host_position, self.host_mean, hosts, self.host_sd, factor)
        for leg, got in guests.items():
            self._place_guests(out, leg, hosts, got, factor)
        return out

    def _decode_paired(self, got, factor, averaged=False):
        """Two-channel tables: each channel has been decoded on its own
        stairs (``soften``); base and detail are separated as the stock
        slices profile does, weighted by what each channel's noise left of
        them."""
        out = np.zeros(LUMA)
        shared = (self.host_position, self.host_mean, self.host_sd, self.common, self.differ,
                  self.ride, self.detail_position, factor)
        if 'sum' in got:
            sk.place_single(out, got['sum'][0], *shared, 0.0)
        elif len(got) == 2:
            (left, _, sigma_left), (right, _, sigma_right) = got[0], got[1]
            sk.place_pair(out, left, right, sigma_left, sigma_right, self.level, self.step,
                          *shared, bool(averaged))
        else:
            (leg, (hosts, _, _)), = got.items()
            sk.place_single(out, hosts, *shared, 1.0 if leg == 0 else -1.0)
        for leg, (hosts, guests, _) in got.items():
            if leg != 'sum':
                self._place_guests(out, leg, hosts, guests, factor)
        return out


def _NOTHING(slots, _made={}):
    """A zero array of ``slots`` values (the unused second guest)."""
    if slots not in _made:
        _made[slots] = np.zeros(slots)
    return _made[slots]


# Held pictures: a dithered packet's stair error differs from packet to
# packet, so the mean of the last DITHER_PHASES decoded pictures of a scene
# that is not moving carries about a seventh of it.  NESTED_FOLD_HOLD=0 shows
# every packet as decoded.
HOLD = os.environ.get('NESTED_FOLD_HOLD', '1') != '0'
# Mean squared change of the plain (unfolded) hosts between two packets, in
# units of what their noise explains, above which the picture has moved.
HOLD_MOTION = 1.5
# The same on the folded hosts, whose stair errors differ between packets;
# a still picture reads about 1 there.
HOLD_STAIR_MOTION = 1.5
# Change of a plain host, as a share of its spread, that is never motion.
HOLD_FLOOR = .003


class Held:
    """Mean of the decoded luma over the dither cycle of a still picture.

    Motion is read on the hosts that are sent plain: they carry no stair, so
    between two packets of one picture they differ by noise alone.  Any more
    than that, a change of level, a packet out of sequence or a packet with
    plain stairs starts again from the packet in hand, which is then shown
    exactly as decoded.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._planes = None            # (DITHER_PHASES, luma) ring of decoded pictures
        self._count = self._next = 0
        self._key, self._phase, self._hosts = None, None, _NONE
        self.last_count = 0
        self.last_motion = None

    def update(self, fold, luma, level, phase, sigma, key=None, averaged=None):
        """``luma``: the packet as decoded; ``averaged()``: the same packet
        decoded for averaging, where that differs (two-channel tables)."""
        if not HOLD or phase is None:
            self.reset()
            return luma
        key = (key, id(fold), level)
        step = None if self._phase is None else (phase-self._phase) % DITHER_PHASES
        still = self._key == key and step in (1, DITHER_PHASES-1)
        if not still or len(self._hosts) != fold.slots:
            self._hosts = np.empty(fold.slots)
            still = False
        # What two packets of one picture differ by: noise, and on folded
        # hosts the two packets' stair errors.
        motion = sk.held_motion(
            luma, self._hosts if still else _NONE, fold.host_position, fold.level, fold.step,
            fold.host_sd, fold.scale, _f8(sigma), RATIO**level, HOLD_FLOOR)
        if still:
            self.last_motion = (float(motion[0]), float(motion[1]))
            still = motion[0] < HOLD_MOTION and motion[1] < HOLD_STAIR_MOTION
        else:
            self._hosts[:] = sk.gather(luma, fold.host_position)
        if not still or self._planes is None or self._planes.shape[1] != len(luma):
            self._planes = np.empty((DITHER_PHASES, len(luma)))
            self._count = self._next = 0
        self._key, self._phase = key, phase
        self._planes[self._next] = luma if averaged is None else averaged()
        self._next = (self._next+1) % DITHER_PHASES
        self._count = min(self._count+1, DITHER_PHASES)
        self.last_count = self._count
        if self._count == 1:
            return luma
        return sk.mean_of_rows(self._planes, self._count)


# Decoding inside the bounds (tools/v7_sk_fold.py, smooth_within_bounds): the
# picture shown is the cleanest one that fits what was received.  A host on a
# staircase may move within half a step, a plainly sent coefficient within its
# noise, a guest not at all, and coefficients that were not sent are free.
# On dense texture with no flat areas this costs a little (about 5% more
# luma error on 1/f noise); on the fixtures it lowers the error.
# SMOOTH_PASSES is the strength (0: show the decoded values as they are);
# the descent carries momentum, and 16 passes of it reach what 40 plain
# passes do;
# NESTED_FOLD_SMOOTH or the receiver's --nested-smooth sets it.
SMOOTH_PASSES = int(os.environ.get('NESTED_FOLD_SMOOTH', 16))
# Share of the half stair a folded host may move.  All of it gives the
# flattest picture; half of it gave the lowest error on the fixtures and
# costs dense texture less.
SMOOTH_ROOM = float(os.environ.get('NESTED_FOLD_SMOOTH_ROOM', .5))
SMOOTH_EDGE = .01              # pixel difference (of -1..1) that counts as flat
SMOOTH_RATE = .25              # descent step, as a share of SMOOTH_EDGE

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
    if variance is not None:
        return sk.slot_noise(float(relative), design, 2, _f8(variance), SLOT_NOISE_SHARE)
    if confidence is not None:
        return sk.slot_noise(float(relative), design, 1, _f8(confidence), SLOT_NOISE_SHARE)
    return sk.slot_noise(float(relative), design, 0, _NONE, SLOT_NOISE_SHARE)


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
        self.dithered = False
        self._phase, self._advance = None, 1

    def phase(self, tail_slice):
        """Dither phase of the packet in hand.  It is the tail slice the
        packet's metadata names (the sender's counter modulo seven); a packet
        whose metadata could not be read is taken to follow the last one."""
        if tail_slice is None:
            if self._phase is None:
                return 0
            tail_slice = self._phase+self._advance
        else:
            tail_slice = int(tail_slice)
            if self._phase is not None:
                advance = (tail_slice-self._phase) % DITHER_PHASES
                if advance in (1, DITHER_PHASES-1):
                    self._advance = advance
        self._phase = tail_slice % DITHER_PHASES
        return self._phase

    def decide(self, score, level, stock_score, dithered=False):
        """(is nested, level) for a packet with these signature readings.
        ``self.dithered`` then says whether its stairs are dithered."""
        if score >= SCORE_MIN and score > stock_score:
            self.kind, self.level, self.age = 'nested', level, 0
            self.dithered = bool(dithered)
        elif stock_score >= SCORE_MIN and stock_score > score:
            self.kind, self.age = 'stock', 0
        else:
            self.age += 1
            if self.age > STICKY_PACKETS:
                self.kind = None
            if self.kind is None:
                # Never seen a readable packet: go by the better reading.
                self.dithered = bool(dithered)
                return score > stock_score and score > .25, level
        return self.kind == 'nested', self.level

    def reset(self):
        self.kind, self.age = None, 0
        self._phase, self._advance = None, 1


def packet_noise(residual, design, own=None):
    """The packet-wide (hiss-like) noise relative to the table's design.

    ``residual``: RMS distance of the signature slots from their pattern, in
    slot signal units; ``design``: the table's design noise there; ``own``:
    the equaliser's own noise figure for those slots.  What the equaliser
    already attributes to those particular slots (they sit in a damaged band,
    say) is not spread over the whole packet.
    """
    return float(sk.packet_noise(float(residual), float(design),
                                 _NONE if own is None else _f8(own), SLOT_NOISE_SHARE))


_SENDING = threading.local()   # sender: counter of the packet being built


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
        # Set-up, once per codec: slices of the stock model in the type and
        # layout the kernels take.
        self.index = np.ascontiguousarray(index, np.int64)
        self.mu = _f8(np.asarray(stock.model.mu)[index])
        self.sd = _f8(np.sqrt(np.asarray(stock.model.lam)[index]))
        self.signature_index = np.ascontiguousarray(
            np.asarray(stock.hosts)[-SIGNATURE_SLOTS:], np.int64)
        self.signature_sd = _f8(np.asarray(stock.sd_host)[-SIGNATURE_SLOTS:])
        self.signature_mu = _f8(np.asarray(stock.model.mu)[self.signature_index])
        self.pattern = _f8(stock.pattern)
        from folding import SIGNATURE_STEPS
        self.signature_unit = SIGNATURE_STEPS*stock.D/math.sqrt(stock.power)
        self._unit_factor = _f8(self.pattern/self.signature_unit)
        self._ones = np.ones(len(self.index))
        self.design_signature_noise = float(sk.root_mean_square(fold.signature_noise))
        self.folded_slots = int(np.count_nonzero(fold.level > 0))
        self.last_level = None
        self.last_nested = False
        self.last_dithered = False
        self.recognition = Recognition()
        self.held = Held()
        self.tail_slice = None          # receiver: tail slice of the packet in hand
        self._level_plane, self._level = _NONE, 0   # sender: last picture and its level

    def _unit(self, xhat, conf):
        seen, confidence = sk.unshrink(_f8(xhat), _f8(conf), 1e-3)
        return confidence, seen, sk.gather_scaled(
            seen, self.signature_index, self.signature_sd, self._unit_factor)

    def begin_packet(self, tail_slice, xhat, conf):
        """Receiver: note the tail slice the packet's metadata names (its
        dither phase).  True when the packet's stairs are dithered: such
        packets differ from one to the next by design, so their slot values
        must not be averaged over packets before the fold is undone."""
        self.tail_slice = tail_slice
        score, _, _, stock_score, dithered = read_signature(self._unit(xhat, conf)[2])
        if score >= SCORE_MIN and score > stock_score:
            return bool(dithered)
        return self.recognition.kind == 'nested' and self.recognition.dithered

    def __getattr__(self, name):
        return getattr(self.stock, name)

    def sent_luma_mask(self):
        return self.fold.sent_mask if self.send else self.stock.sent_luma_mask()

    def encode_coefficients(self, values, full=None):
        if full is None:
            full = self.stock.grid.forward(values)
        coefficients = np.array(self.stock.encode_coefficients(values, full), np.float64)
        if not self.send:
            return coefficients
        plane = _f8(full)[:LUMA]
        # A held picture is sent packet after packet: its level is chosen once.
        if not sk.same_values(plane, self._level_plane):
            self._level_plane, self._level = plane.copy(), self.fold.choose_level(plane)
        level = self._level
        # Dithered only inside a packet whose counter is known (see
        # enable_mono); a bare call sends plain stairs.
        counter = getattr(_SENDING, 'counter', None) if DITHER else None
        phase = None if counter is None else int(counter) % DITHER_PHASES
        coefficients[self.index] = sk.slot_coefficients(
            self.mu, self.sd, self.fold.encode(plane, level, 0, phase))
        coefficients[self.signature_index] = sk.signature_symbols(
            self.signature_mu, self.signature_sd, self.signature_unit, self.pattern,
            HADAMARD[int(level)-LOWEST+1], -1.0 if phase is not None else 1.0)
        return coefficients

    def decode(self, coeffs, xhat, conf, fallback=True, metadata_confirmed=False):
        confidence, seen, unit = self._unit(xhat, conf)
        score, level, residual, stock_score, dithered = read_signature(unit)
        self.last_nested, level = self.recognition.decide(
            score, level, stock_score, dithered)
        tail_slice, self.tail_slice = self.tail_slice, None
        if not self.last_nested:
            self.held.reset()
            return self.stock.decode(coeffs, xhat, conf, fallback=fallback,
                                     metadata_confirmed=metadata_confirmed)
        fold = self.fold
        relative = packet_noise(residual*self.signature_unit, self.design_signature_noise,
                                sk.confidence_noise(confidence, self.signature_index))
        sigma = slot_noise(relative, fold.noise, sk.gather(confidence, self.index))
        full = self.stock.plain(coeffs)
        self.last_dithered = self.recognition.dithered
        phase = self.recognition.phase(tail_slice) if self.last_dithered else None
        luma = self.held.update(
            fold, fold.decode({0: (sk.gather_scaled(seen, self.index, self.sd, self._ones),
                                   sigma)}, level, phase),
            level, phase, sigma)
        full[:LUMA] = fold.clean(luma, fold.room(level, sigma, self.held.last_count))
        self.stock.last_score = self.last_score = score
        self.stock.last_noise = self.last_noise = relative
        self.stock.last_unfolded_slots = self.last_unfolded_slots = self.folded_slots
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
            nested.held.reset()

    wire._codec = codec
    wire.nested_fold = 'send' if send else 'receive'
    wire.reset_nested = forget          # a new stream: forget what the last one carried
    if send:
        # Nested packets go out auto-levelled: the fold leaves some pictures'
        # bodies well under the ceiling, and that headroom is signal-to-noise.
        from animation_modem import v7
        encode_packet = wire.encode_packet

        def levelled(model, values, counter, *args, **kwargs):
            _SENDING.counter = counter
            try:
                with v7.body_auto_level():
                    return encode_packet(model, values, counter, *args, **kwargs)
            finally:
                _SENDING.counter = None

        wire.encode_packet = levelled
    return wire


def _slice_wire_class():
    import slice_wire as stock

    class NestedHalf(stock.Half):
        __slots__ = ('signature', 'counter')

    class NestedSliceWire(stock.SliceWire):
        """Stereo slices whose channels are nested-fold mono pictures with
        shared hosts and different guests.  Reads stock slices unchanged."""

        def __init__(self, layout='auto', send=False, path=None):
            super().__init__(layout)
            self.send, self._path = bool(send), path
            self._level = (None, None)
            self._counter = None
            self._prepared = {}
            self.nested_fold = 'send' if send else 'receive'
            self.last_nested = False
            self.recognition = Recognition()
            self.held = Held()

        def reset(self):
            super().reset()
            self.recognition.reset()
            self.held.reset()

        def fold(self, layout):
            fold = table('slices', layout, self._path)
            if fold is not None and not np.array_equal(
                    fold.host_position, self.slots(layout).base[0]):
                raise ValueError('nested fold table does not match this layout')
            return fold

        def prepared(self, layout):
            """Set-up, once per layout: slices of the stock slot tables in
            the type and layout the kernels take."""
            if layout not in self._prepared:
                slots, fold = self.slots(layout), self.fold(layout)
                part = np.ascontiguousarray(np.arange(len(slots.model_mu))[slots.parts[0]], np.int64)
                hosts = np.ascontiguousarray(slots.fold_hosts[slots.guests:], np.int64)
                amplitude = stock.SIGNATURE_STEPS*stock.FOLD_STEP/math.sqrt(stock.FOLD_POWER)
                pattern = _f8(stock.SIGNATURE_PATTERN)
                transforms = []
                for rows, cols in stock.v7.V7_GRIDS:
                    transforms.append((np.ascontiguousarray(stock._dct_matrix(rows).T, np.float64),
                                       _f8(stock._dct_matrix(cols))))
                self._prepared[layout] = types.SimpleNamespace(
                    part=part, part_mu=_f8(slots.model_mu[part]),
                    part_sd=_f8(np.sqrt(slots.model_lam[part])),
                    hosts=hosts, host_mu=_f8(slots.model_mu[hosts]),
                    host_sd=_f8(slots.sd_host[slots.guests:]),
                    amplitude=amplitude, pattern=pattern,
                    unit_factor=_f8(pattern/amplitude), model_mu=_f8(slots.model_mu),
                    lam=_f8(slots.lam[0]),
                    design=max(float(sk.root_mean_square(fold.signature_noise)), 1e-12)
                    if fold is not None else 1.0,
                    transforms=transforms,
                    base=[np.ascontiguousarray(index, np.int64) for index in slots.base],
                    pixels=sum(rows*cols for rows, cols in stock.v7.V7_GRIDS))
            return self._prepared[layout]

        def luma_sent_mask(self, layout):
            fold = self.fold(layout) if self.send else None
            return super().luma_sent_mask(layout) if fold is None else fold.sent_mask

        # ------------------------------------------------------------ sender
        def encode_packet(self, base_model, values, counter, *args, **kwargs):
            # Nested packets go out auto-levelled (see enable_mono).
            self._counter = counter
            try:
                with stock.v7.body_auto_level(self.send):
                    return super().encode_packet(base_model, values, counter,
                                                 *args, **kwargs)
            finally:
                self._counter = None

        def channel_coefficients(self, model, spectra, kind):
            out = super().channel_coefficients(model, spectra, kind)
            layout = self._layout_of[id(model)]
            fold = self.fold(layout) if self.send else None
            if fold is None:
                return out
            ready = self.prepared(layout)
            plane = spectra[0]
            # Both channels of a frame share one level: choose it once.
            if self._level[0] is not plane:
                self._level = (plane, fold.choose_level(plane))
            level = self._level[1]
            phase = (int(self._counter) % DITHER_PHASES
                     if DITHER and self._counter is not None else None)
            symbols = fold.encode(plane, level, stock.SIDES.index(kind), phase)
            out[ready.part] = sk.slot_coefficients(ready.part_mu, ready.part_sd, symbols)
            out[ready.hosts] = sk.signature_symbols(
                ready.host_mu, ready.host_sd, ready.amplitude, ready.pattern,
                HADAMARD[int(level)-LOWEST+1], -1.0 if phase is not None else 1.0)
            return out

        # ---------------------------------------------------------- receiver
        def half(self, model, result, channel=None):
            plain = super().half(model, result, channel)
            ready = self.prepared(plain.layout)
            equalized = result.diag.get('mono_fold_eq')
            if equalized is not None:
                seen, trust = sk.unshrink(_f8(equalized[0]), _f8(equalized[1]), 1e-3)
            else:
                seen = sk.subtract(_f8(result.coeffs), ready.model_mu)
            unit = sk.gather_scaled(seen, ready.hosts, ready.host_sd, ready.unit_factor)
            half = NestedHalf(plain.kind, plain.score, plain.layout, plain.good, plain.noise,
                              plain.damaged, plain.steady, plain.raw, plain.ring)
            own = None
            if equalized is not None:
                own = sk.confidence_noise(trust, ready.hosts)
            half.signature = read_signature(unit)+(own,)
            half.counter = result.diag.get('tail_slice')
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
                self.last_nested, level = self.recognition.decide(
                    best[0], best[1], best[3], best[4])
            if not self.last_nested:
                self.held.reset()
                return super().values(halves)
            ready = self.prepared(chosen[0].layout)
            lam, unit_scale, design = ready.lam, ready.amplitude, ready.design
            kinds = [half.kind for half in chosen]
            leak = self.nested_crosstalk(chosen) if sorted(kinds) == sorted(stock.SIDES) else 0.0
            seen = {half.kind: _f8(half.raw[0][0]) for half in chosen}
            if leak:
                a, b = sk.undo_crosstalk(seen['left'], seen['right'], float(leak))
                seen = {'left': a, 'right': b}
            readings = {}
            for half, (_, _, residual, _, _, own) in zip(chosen, signed):
                values, variance = sk.scaled_reading(seen[half.kind], _f8(half.raw[0][1]), lam)
                sigma = slot_noise(packet_noise(residual*unit_scale, design, own), fold.noise,
                                   variance=variance)
                key = 'sum' if half.kind == 'sum' else stock.SIDES.index(half.kind)
                readings[key] = (values, sigma)
            if 'sum' in readings:
                readings = {'sum': readings['sum']}
            named = [half.counter for half in chosen if half.counter is not None]
            phase = (self.recognition.phase(named[0] if named else None)
                     if self.recognition.dithered else None)
            soft = fold.soften(readings, phase)
            luma = self.held.update(
                fold, fold.decode(readings, level, phase, soft=soft), level, phase,
                next(iter(readings.values()))[1],
                key=(chosen[0].layout, tuple(sorted(map(str, readings)))),
                averaged=lambda: fold.decode(readings, level, phase, averaged=True, soft=soft))
            worst = None
            for _, sigma in readings.values():
                worst = sigma if worst is None else sk.larger(worst, sigma)
            luma = fold.clean(luma, fold.room(level, worst, self.held.last_count,
                                              joined=len(readings) == 2))
            out = np.empty(ready.pixels)
            start = 0
            for plane, ((base, _), (rows, cols)) in enumerate(
                    zip(self.estimate(chosen), stock.v7.V7_GRIDS)):
                spectrum = _f8(luma) if plane == 0 else sk.scatter(
                    rows*cols, ready.base[plane], _f8(base))
                start = sk.inverse_plane(spectrum, out, start, *ready.transforms[plane])
            return out

    return NestedSliceWire


def slice_wire(layout='auto', send=False, path=None):
    """A stereo-slices wire that reads nested-fold packets (and sends them
    with ``send``)."""
    return _slice_wire_class()(layout, send=send, path=path)
