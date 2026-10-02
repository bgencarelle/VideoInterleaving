"""2:1 luma folding around the unchanged V7 modem.

The M weakest luma body slots each carry two luma coefficients. The slot's own
coefficient (the host) is sent coarsely, as a multiple of a step D. The most
important luma coefficient V7 does not send today (the guest, from the 96x80
luma grid outside the 48x40 corner) rides inside the step as a small analog
residual:

    s = D*round(h/D) + beta*clip(u, -U, U)      h, u unit-variance host, guest

The folded symbol replaces the host coefficient at the host's own variance,
so the wire, its power and the equaliser are unchanged: a normal V7 encode of
the modified coefficients is a folded packet. The receiver takes the
equaliser's per-coefficient estimate and confidence for the host slots,
removes the MMSE shrink, and splits the symbol back into host and guest.

Fallback: a host whose equaliser confidence is below conf_min is read as a
plain noisy host, and its guest is dropped. This bounds the damage when timing
errors (fast flutter, jitter) push a symbol over a step.

Companded guests (use_compand): real pictures' guests are heavy-tailed and
several times the model's variance (measured 1.5 to 5 times its standard
deviation), so the +-2.5 clip above throws most of their energy away (guest
SDR 1.8 dB on a clean channel for a frame of text on black). With companding
the guest rides as

    s = D*round(h/D) + A*c(u),   c(u) = sign(u)*ln(1 + mu*|u|/U)/ln(1 + mu),

|u| limited to U and A = 0.4*D, so a guest up to U standard deviations
survives and small guests keep nearly the same slope. The receiver shrinks
c by its MMSE weight at the measured symbol noise, then expands. Past
guest_noise_max (symbol units) the guests are dropped but the hosts are
still read as quantised steps, which D makes robust; the display's edge
reconstruction treats dropped guests as unknown and fills them in.
"""
import hashlib
import json
from contextlib import contextmanager

import numpy as np

from common import Grids, fit_variances, v7, v7_values

U_CLIP = 2.5
SIGNATURE_STEPS = 3
NOISE_MAX = .3          # signature symbol noise (RMS, in steps) above which a packet is not unfolded
TABLE_FORMAT = 'v7-fold-2'


def model_digest(model):
    """Identity of a source model's statistics. A fold table fixes hosts,
    guests and scales for one model (encode filter and profile); another
    model must not be folded or unfolded with it."""
    digest = hashlib.sha256(f'encoding {int(model.encoding_type)}'.encode())
    for array in (model.mu, model.lam, model.gain):
        digest.update(np.ascontiguousarray(array, dtype='<f8').tobytes())
    digest.update(np.ascontiguousarray(model.order, dtype='<i8').tobytes())
    return digest.hexdigest()


def table_text(table):
    """The one serialisation of a fold table: its file bytes and its identity."""
    return json.dumps(table, sort_keys=True, separators=(',', ':')) + '\n'


def table_identity(table):
    return hashlib.sha256(table_text(table).encode()).hexdigest()


class FoldCodec:
    def __init__(self, model, train_frames, M, D=None, design_db=30.0,
                 encode_filter='box', conf_min=.9, signature=0, fitted_on=''):
        self.model, self.M, self.filter, self.conf_min = model, int(M), encode_filter, conf_min
        self.design_db, self.fitted_on = float(design_db), str(fitted_on)
        self.signature, self.noise_max = int(signature), NOISE_MAX
        self.grid = Grids(v7.V7_GRIDS)
        self.kept = self.grid.corner_positions(v7.V7_SHAPES)            # model coefficient -> grid
        luma_end = int(np.prod(v7.V7_SHAPES[0]))
        _, lam = fit_variances(self.grid, lambda im: v7_values(im, encode_filter), train_frames)
        rank = np.empty(len(model.order), int)
        rank[model.order] = np.arange(len(model.order))
        body = np.flatnonzero((np.arange(len(rank)) < luma_end) &
                              (rank >= v7.HEAD) & (rank < v7.BODY_END))
        body = body[np.argsort(rank[body])]                             # most important first
        luma_grid = self.grid.off[1]
        outside = np.setdiff1d(np.arange(luma_grid), self.kept[:luma_end])
        outside = outside[np.argsort(-lam[outside], kind='stable')]
        self.hosts = body[-self.M:]                                      # weakest luma body slots
        self.guests = outside[:self.M]                                   # next luma coefficients
        self.sd_host = np.sqrt(model.lam[self.hosts])
        self.sd_guest = np.sqrt(lam[self.guests])
        self.train = train_frames
        self.set_step(D if D is not None else self.best_step(design_db))
        self._set_identity()

    def _set_identity(self):
        """The weakest `signature` hosts carry a fixed +-SIGNATURE_STEPS*D
        pattern instead of data, so a receiver can tell a folded packet from a
        normal one without a packet flag (unfolding a normal packet is far
        worse than not unfolding a folded one). The pattern is drawn from the
        table's identity, so a receiver holding a different table does not
        recognise the packet and shows it unfolded instead of mis-unfolding it."""
        self.identity = table_identity(self.table())
        rng = np.random.default_rng(int(self.identity[:16], 16))
        self.pattern = rng.choice([-1.0, 1.0], self.signature)
        self.last_score = self.last_noise = None
        self.last_unfolded_slots = 0

    # ----------------------------------------------------------- fold table
    # Sender and receiver must fold the same slots with the same scales, so a
    # live pair loads one frozen table instead of fitting local frames.
    compand = None                  # (limit U, mu) or None: linear guests
    guest_noise_max = None
    COMPAND_POWER = .12             # E c(u)^2 for real guests (measured .10-.14)

    def use_compand(self, limit, mu, step, guest_noise_max):
        """Switch to companded guests (changes the table identity and the
        signature: both ends must hold the same settings)."""
        self.compand = (float(limit), float(mu))
        self.guest_noise_max = float(guest_noise_max)
        self.set_step(step)
        self._set_identity()
        return self

    def _compress(self, u):
        limit, mu = self.compand
        return (np.sign(u)*np.log1p(mu*np.minimum(np.abs(u), limit)/limit) /
                np.log1p(mu))

    def _expand(self, c):
        limit, mu = self.compand
        return np.sign(c)*limit*np.expm1(np.abs(c)*np.log1p(mu))/mu

    def table(self):
        table = self._table()
        if self.compand is not None:
            table.update({'compand': list(self.compand),
                          'guest_noise_max': self.guest_noise_max})
        return table

    def _table(self):
        return {'format': TABLE_FORMAT, 'M': self.M, 'D': self.D, 'u_clip': U_CLIP,
                'conf_min': self.conf_min, 'encode_filter': self.filter,
                'model_digest': model_digest(self.model),
                'model_tables_sha256': v7.MODEL_TABLES_SHA256,
                'design_db': self.design_db, 'fitted_on': self.fitted_on,
                'signature': self.signature, 'noise_max': self.noise_max,
                'hosts': self.hosts.tolist(), 'guests': self.guests.tolist(),
                'sd_guest': self.sd_guest.tolist()}

    @classmethod
    def from_table(cls, model, table):
        """A codec for `model` from a frozen table. Fails closed: the table must
        be this format and must have been built for exactly this model."""
        if table.get('format') != TABLE_FORMAT or table.get('u_clip') != U_CLIP:
            raise ValueError(f'unsupported fold table (need format {TABLE_FORMAT})')
        if table.get('model_digest') != model_digest(model):
            raise ValueError(
                f"fold table was built for the canonical '{table.get('encode_filter')}' "
                f"profile, not this model (encode filter "
                f"'{v7.ENCODING_FILTERS[int(model.encoding_type)]}'"
                f"{' or a custom fixture' if table.get('encode_filter') == v7.ENCODING_FILTERS[int(model.encoding_type)] else ''})")
        codec = cls.__new__(cls)
        codec.model, codec.M = model, int(table['M'])
        codec.filter = table['encode_filter']
        codec.conf_min = float(table['conf_min'])
        codec.design_db, codec.fitted_on = float(table['design_db']), table['fitted_on']
        codec.grid = Grids(v7.V7_GRIDS)
        codec.kept = codec.grid.corner_positions(v7.V7_SHAPES)
        codec.hosts = np.asarray(table['hosts'], int)
        codec.guests = np.asarray(table['guests'], int)
        if not (len(codec.hosts) == len(codec.guests) == len(table['sd_guest']) == codec.M):
            raise ValueError('fold table lengths disagree')
        codec.sd_host = np.sqrt(model.lam[codec.hosts])
        codec.sd_guest = np.asarray(table['sd_guest'], float)
        codec.train = []
        codec.signature = int(table['signature'])
        codec.noise_max = float(table['noise_max'])
        codec.set_step(float(table['D']))
        if table.get('compand') is not None:
            codec.use_compand(*table['compand'], float(table['D']),
                              table['guest_noise_max'])
        codec._set_identity()
        if codec.identity != table_identity(table):
            raise ValueError('fold table does not round-trip')          # pragma: no cover
        return codec

    # Model indices of the kept coefficients every packet carries; None means
    # all of them. Profiles that send a fixed subset (mono) override it.
    sent_model_indices = None

    def sent_luma_mask(self):
        """Luma-grid mask of the coefficients this table carries (kept + guests)."""
        cached = getattr(self, '_sent_luma_mask', None)
        if cached is not None:
            return cached
        luma = int(self.grid.off[1])
        kept = (self.kept if self.sent_model_indices is None else
                self.kept[np.asarray(self.sent_model_indices, dtype=int)])
        mask = np.zeros(luma, dtype=bool)
        mask[kept[kept < luma]] = True
        guests = np.asarray(self.guests)
        mask[guests[guests < luma]] = True
        mask.setflags(write=False)
        self._sent_luma_mask = mask
        return mask

    def set_step(self, D):
        self.D = float(D)
        if self.compand is not None:
            self.amp = .4*self.D
            self.power = 1 + self.D**2/12 + self.amp**2*self.COMPAND_POWER
            return
        self.beta = .8*self.D/(2*U_CLIP)
        self.power = 1 + self.D**2/12 + self.beta**2                    # E s^2

    # ---------------------------------------------------------------- sender
    def _split(self, values):
        full = self.grid.forward(values)
        coeffs = full[self.kept].copy()
        h = (coeffs[self.hosts] - self.model.mu[self.hosts])/self.sd_host
        u = np.clip(full[self.guests]/self.sd_guest, -U_CLIP, U_CLIP)
        return full, coeffs, h, u

    def encode_coefficients(self, values, full=None):
        """Return folded source coefficients, before the inverse DCT to pixels.
        ``full``: the grid's forward transform of ``values``, when the caller
        already has it."""
        if self.compand is not None:
            if full is None:
                full = self.grid.forward(values)
            coeffs = full[self.kept].copy()
            h = (coeffs[self.hosts] - self.model.mu[self.hosts])/self.sd_host
            symbol = (self.D*np.round(h/self.D) +
                      self.amp*self._compress(full[self.guests]/self.sd_guest))
        else:
            _, coeffs, h, u = self._split(values)
            symbol = self.D*np.round(h/self.D) + self.beta*u
        if self.signature:
            symbol[-self.signature:] = SIGNATURE_STEPS*self.D*self.pattern
        coeffs[self.hosts] = self.model.mu[self.hosts] + self.sd_host*symbol/np.sqrt(self.power)
        return coeffs

    def encode(self, values):
        """V7 values whose coefficients carry the folded symbols."""
        return v7.values_from(self.model, self.encode_coefficients(values))

    # -------------------------------------------------------------- receiver
    def _unfold(self, symbol):
        k = np.round(symbol/self.D)
        return self.D*k, np.clip((symbol - self.D*k)/self.beta, -U_CLIP, U_CLIP)

    def decode(self, coeffs, xhat, conf, fallback=True,
               metadata_confirmed=False):
        """Full 96x80/48x40 DCT vector from a decoded packet: `coeffs` is the
        decoder's result, `xhat`/`conf` the equaliser output for that packet."""
        full = np.zeros(self.grid.off[-1])
        full[self.kept] = coeffs
        self.last_noise = None
        self.last_unfolded_slots = 0
        c = conf[self.hosts]
        symbol = xhat[self.hosts]/np.maximum(c, 1e-3)/self.sd_host*np.sqrt(self.power)
        signature_detected = not self.signature
        if self.signature:
            # ~1 on a folded packet; ~0 +- .09 on a normal one (16 unit-variance
            # hosts against a +-3D pattern).
            self.last_score = float(np.mean(symbol[-self.signature:]*self.pattern) /
                                    (SIGNATURE_STEPS*self.D))
            signature_detected = self.last_score >= .5
            if not signature_detected and not metadata_confirmed:
                return full                          # a normal packet: show it as is
        if self.compand is not None:
            return self._decode_companded(full, coeffs, symbol, c, fallback,
                                          signature_detected,
                                          metadata_confirmed)
        h, u = self._unfold(symbol)
        ok = c >= self.conf_min if fallback else np.ones(self.M, bool)
        if self.signature and fallback and signature_detected:
            # The known signature symbols measure this packet's symbol noise on
            # exactly the folded slots. Timing smear (fast flutter, jitter) is
            # not visible in the equaliser confidence, but it is visible here:
            # past a fraction of the step, unfolding costs more than it gives.
            resid = symbol[-self.signature:] - SIGNATURE_STEPS*self.D*self.pattern
            self.last_noise = float(np.sqrt(np.mean(resid**2))/self.D)
            if self.last_noise > self.noise_max:
                ok[:] = False
            # MMSE weight for the guests at the measured symbol noise.
            noise = (self.last_noise*self.D)**2
            u = u*self.beta**2/(self.beta**2 + noise)
        mu = self.model.mu[self.hosts]
        plain = mu + (np.asarray(coeffs)[self.hosts] - mu)*np.sqrt(self.power)
        full[self.kept[self.hosts]] = np.where(ok, mu + self.sd_host*h, plain)
        full[self.guests] = np.where(ok, u*self.sd_guest, 0.0)
        if self.signature and (signature_detected or metadata_confirmed):
            # Signature slots are reserved in every folded packet even when
            # mono/noise hides their identity. A valid coded mode can authorize
            # the table while the normal host-confidence fallback remains on.
            full[self.kept[self.hosts[-self.signature:]]] = mu[-self.signature:]
            full[self.guests[-self.signature:]] = 0.0
        self.last_unfolded_slots = int(np.count_nonzero(ok))
        return full

    def _decode_companded(self, full, coeffs, symbol, c, fallback,
                          signature_detected, metadata_confirmed):
        steps = np.round(symbol/self.D)
        residual = np.clip((symbol-self.D*steps)/self.amp, -1.0, 1.0)
        ok = c >= self.conf_min if fallback else np.ones(self.M, bool)
        guests_ok = ok.copy()
        noise = 0.0
        if self.signature and fallback and signature_detected:
            resid = (symbol[-self.signature:] -
                     SIGNATURE_STEPS*self.D*self.pattern)
            sigma = float(np.sqrt(np.mean(resid**2)))
            self.last_noise = sigma/self.D
            noise = sigma*sigma
            if self.last_noise > self.noise_max:
                ok[:] = False                    # steps would slip: plain hosts
            if sigma > self.guest_noise_max:
                guests_ok[:] = False             # guests are mostly noise
        guests_ok &= ok
        carried = self.amp**2*self.COMPAND_POWER
        u = self._expand(residual*carried/(carried+noise))
        mu = self.model.mu[self.hosts]
        plain = mu + (np.asarray(coeffs)[self.hosts] - mu)*np.sqrt(self.power)
        full[self.kept[self.hosts]] = np.where(ok, mu + self.sd_host*self.D*steps,
                                               plain)
        full[self.guests] = np.where(guests_ok, u*self.sd_guest, 0.0)
        if self.signature and (signature_detected or metadata_confirmed):
            full[self.kept[self.hosts[-self.signature:]]] = mu[-self.signature:]
            full[self.guests[-self.signature:]] = 0.0
        self.last_unfolded_slots = int(np.count_nonzero(ok))
        return full

    def signature_score(self, xhat, conf):
        """About 1 when the equaliser output carries this table's signature,
        about 0 otherwise (None without a signature). Lets a receiver confirm
        a layout it could not read from the packet's metadata."""
        if not self.signature:
            return None
        hosts = self.hosts[-self.signature:]
        symbol = (np.asarray(xhat)[hosts]/np.maximum(np.asarray(conf)[hosts], 1e-3) /
                  self.sd_host[-self.signature:]*np.sqrt(self.power))
        return float(np.mean(symbol*self.pattern)/(SIGNATURE_STEPS*self.D))

    def signature_error(self, xhat, conf):
        """RMS distance of the signature slots from this table's pattern, in
        units of the pattern's own size: near 0 when it is there, 1 or more
        when the slots hold something else (None without a signature). Unlike
        the score, large unrelated values cannot pass it by chance."""
        if not self.signature:
            return None
        hosts = self.hosts[-self.signature:]
        symbol = (np.asarray(xhat)[hosts]/np.maximum(np.asarray(conf)[hosts], 1e-3) /
                  self.sd_host[-self.signature:]*np.sqrt(self.power))
        size = SIGNATURE_STEPS*self.D
        return float(np.sqrt(np.mean((symbol-size*self.pattern)**2))/size)

    def plain(self, coeffs):
        """Full DCT vector of an unfolded (normal) decode."""
        full = np.zeros(self.grid.off[-1])
        full[self.kept] = coeffs
        return full

    # ---------------------------------------------------------------- design
    def expected_error(self, D, snr_db, seed=5):
        """Host + guest squared error on the training frames, per-slot AWGN at
        a channel SNR where the mean slot power is 1 (V7's gain scaling)."""
        saved = self.D
        self.set_step(D)
        rng = np.random.default_rng(seed)
        slot_power = self.model.gain[self.hosts]**2*self.model.lam[self.hosts]
        sigma = 10**(-snr_db/20)/np.sqrt(slot_power)
        err = 0.0
        for image in self.train:
            full, _, h, u = self._split(v7_values(image, self.filter))
            symbol = self.D*np.round(h/self.D) + self.beta*u
            hh, uu = self._unfold(symbol + sigma*np.sqrt(self.power)*rng.standard_normal(self.M))
            err += (np.sum(((hh - h)*self.sd_host)**2) +
                    np.sum((uu*self.sd_guest - full[self.guests])**2))
        self.set_step(saved)
        return err

    def best_step(self, snr_db):
        self.set_step(1.0)
        return min(np.geomspace(.1, 8, 28), key=lambda D: self.expected_error(D, snr_db))


@contextmanager
def equaliser_capture():
    """Record (xhat, conf) of every packet the V7 decoder equalises."""
    captured = []
    real = v7._equalize_numba

    def capture(model, Z, H, noise, counter):
        out = real(model, Z, H, noise, counter)
        captured.append((out[0].copy(), out[1].copy()))
        return out

    v7._equalize_numba = capture
    try:
        yield captured
    finally:
        v7._equalize_numba = real
