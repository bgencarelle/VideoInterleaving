"""Install the experimental geometric serpentine codec on nested-fold tables.

Preserves frozen table normalization and level parameters. This mapping is
wire-incompatible with legacy V7 and must be installed at both ends.
"""
import numpy as np
from pathlib import Path
import json
import math
import os
import threading


MAPPING = 'serpentine-geometric'
VERSION = 1
_ADAPTIVE = threading.local()
# Receiver: posterior-mean decode at each slot's measured noise
# (v7_sk_fold.serpentine_soft_decode_frame), as the stock nested fold
# decodes. V7_SERPENTINE_SOFT=0 restores the plain geometric read (tests).
SOFT_DECODE = os.environ.get('V7_SERPENTINE_SOFT', '1') != '0'


def install(out=None, fixed_level=None, step_factor=1.0, guest_factor=1.0,
            exact_inverse=False, adaptive_guest_range=False,
            isolated_path=None):
    """Install codec; optionally write metadata and pin sender level."""
    import sys
    repo = Path(__file__).resolve().parents[1]
    test_modem = repo / 'test_modem_v7'
    if str(test_modem) not in sys.path:
        sys.path.insert(0, str(test_modem))
    import nested_fold as nf

    step_factor, guest_factor = float(step_factor), float(guest_factor)
    if not math.isfinite(step_factor) or not 0 < step_factor <= 4:
        raise ValueError('step factor must be finite and in (0, 4]')
    if not math.isfinite(guest_factor) or guest_factor < 1:
        raise ValueError('guest factor must be finite and at least 1')
    adaptive_guest_range = bool(adaptive_guest_range)
    from tools.v7_adaptive_pipeline import ADAPTIVE_HOST_LEVEL
    host_level = int(ADAPTIVE_HOST_LEVEL)
    # The level the stairs are cut at: the soft decode's host prior is on
    # the unnormalised host, which that level divides by RATIO**level.
    host_factor = float(nf.RATIO)**(host_level if fixed_level is None
                                    else int(fixed_level))
    if adaptive_guest_range and (fixed_level != host_level or guest_factor != 1.0):
        raise ValueError(f'adaptive guest range fixes host level {host_level} '
                         'and static guest factor 1')
    configuration = (fixed_level, step_factor, guest_factor, bool(exact_inverse),
                     adaptive_guest_range)
    isolated_key = None
    if isolated_path is not None:
        isolated_key = str(Path(isolated_path).resolve())
        registry = getattr(nf, '_isolated_serpentine_registry', {})
        previous = registry.get(isolated_key)
        if previous is not None:
            if previous != configuration:
                raise ValueError('isolated table already has another serpentine configuration')
            return nf

    if isolated_path is None and nf._cache:
        raise RuntimeError('install serpentine probe before any target fold is cached')
    if isolated_path is None and getattr(nf, '_serpentine_codec_installed', False):
        if nf._serpentine_codec_configuration != configuration:
            raise ValueError('serpentine codec is already installed with different parameters')
        return nf
    from tools import v7_sk_fold as sk

    required = ('serpentine_encode_frame', 'serpentine_decode_frame',
                'serpentine_stair_values', 'serpentine_place_pair',
                'adaptive_guest_peak', 'adaptive_guest_factor_from_peak',
                'adaptive_pilot_pattern')
    if any(not hasattr(sk, name) for name in required):
        raise RuntimeError('serpentine kernels are not available yet')
    decoder = sk.serpentine_decode_frame_exact if exact_inverse else sk.serpentine_decode_frame

    if fixed_level is not None and not nf.LOWEST <= fixed_level <= nf.HIGHEST:
        raise ValueError(f'fold level must be between {nf.LOWEST} and {nf.HIGHEST}')
    metadata_path = None
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        metadata_path = out / 'serpentine-codec.json'

    def record_case(record):
        if metadata_path is None or not metadata_path.exists():
            return
        with metadata_path.open(encoding='utf-8') as stream:
            data = json.load(stream)
        family = record['family']
        key = f"{record['rows']}x{record['cols']}"
        cases = data.setdefault('families', {}).setdefault(family, {})
        cases[key] = record
        with metadata_path.open('w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write('\n')

    # Compile all kernels during installation, before any live frame is handled.
    tiny = np.zeros(1, dtype=np.float64)
    level = np.zeros(1, dtype=np.int64)
    ones = np.ones(1, dtype=np.float64)
    table = np.linspace(0.0, 1.0, 1025)
    inverse = np.linspace(0.0, sk.B_LIMIT, 1025)
    sk.serpentine_encode_frame(tiny, tiny, level, ones, ones, table, tiny)
    decoder(tiny, level, ones, ones, table if exact_inverse else inverse, tiny)
    sk.serpentine_stair_values(tiny, level, ones, tiny)
    sk.serpentine_stair_centroids(tiny, level, ones, tiny, 1.0)
    sk.serpentine_soft_decode_frame(tiny, level, ones, ones, ones,
                                    np.ones(sk.SOFT_POINTS), table, tiny, 1.0)
    sk.serpentine_place_pair(np.zeros(2), tiny, tiny, np.array([0]), tiny, ones,
                             tiny, np.array([1]), 1.0)
    if adaptive_guest_range:
        peak, status = sk.adaptive_guest_peak(
            tiny, np.array([0], dtype=np.int64), ones, ones, nf.RATIO**-1)
        if status:
            raise RuntimeError('adaptive guest selector warm-up failed')
        sk.adaptive_guest_factor_from_peak(peak)
        sk.adaptive_pilot_pattern(nf.HADAMARD, 1, False)

    if isolated_path is None:
        fold_class = nf.FrozenFold
    else:
        fold_class = type('SerpentineFrozenFold', (nf.FrozenFold,), {})
    original_init = fold_class.__init__

    def init(self, arrays):
        # Private setup arrays: keep the file and caller-owned tables intact.
        arrays = dict(arrays)
        arrays['step'] = np.ascontiguousarray(arrays['step']*step_factor, np.float64)
        arrays['guest_sd'] = np.ascontiguousarray(arrays['guest_sd']*guest_factor, np.float64)
        original_init(self, arrays)
        self.adaptive_guest_range = adaptive_guest_range
        if (len(self.compander) < 2 or not np.all(np.isfinite(self.compander))
                or self.compander[0] != 0.0 or self.compander[-1] != 1.0
                or np.any(np.diff(self.compander) <= 0.0)):
            raise ValueError('serpentine compander must be strictly increasing from 0 to 1')
        if np.any((self.level != 0) & (self.level != 1)):
            raise ValueError('serpentine pilot supports frozen levels 0 and 1 only')
        if not self.paired and self.guest_position.shape[0] != 1:
            raise ValueError('serpentine mono pilot requires exactly one guest leg')
        # alpha remains present for table compatibility; this mapping does not use it.
        self._serpentine_zero = np.zeros(self.slots, dtype=np.float64)
        self._serpentine_plain_level = np.zeros(self.slots, dtype=np.int64)
        self._serpentine_inverse = np.ascontiguousarray(self.inverse, dtype=np.float64)
        self._serpentine_decoder_table = (self.compander if exact_inverse
                                         else self._serpentine_inverse)
        self.mapping_metadata = {
            'family': 'slices' if self.paired else 'mono',
            'rows': self.rows, 'cols': self.cols,
            'Kcounts': {'level_0': int(np.count_nonzero(self.level == 0)),
                        'level_1': int(np.count_nonzero(self.level == 1))},
            'mapping': MAPPING, 'version': VERSION,
            'compander': 'frozen table compander; legacy alpha unused',
            'normalization': 'inherited frozen-fold normalization/step/scale',
            'step_factor': step_factor, 'guest_factor': guest_factor,
            'adaptive_guest_range': adaptive_guest_range,
            'pilot_semantics': ('guest factors [1,2,4,8,16] at rows [-3,-2,-1,0,1]; host level -1'
                                if adaptive_guest_range else 'host ladder level'),
            'inverse': 'exact forward-table inversion' if exact_inverse else 'uniform inverse LUT',
            'required_at_both_ends': True,
            'dither': 'uses supplied kernel offsets; pilot harness must pin level',
            'paired': ('full algebraic M/S legs; no covariance shrink' if self.paired
                       else 'single channel'),
            'single_leg': ('stock M/S interpretation' if self.paired else 'not applicable (mono)'),
            'automatic_level_selection': ('fixed level' if fixed_level is not None else 'disabled'),
            'fixed_level': fixed_level,
            'wire_id': None,
            'compatibility': 'requires this mapping at both ends; legacy receivers are unsupported',
            'noise_claim': 'clean geometric inverse; no noise-robustness claim',
        }
        record_case(self.mapping_metadata)

    def stairs(self, hosts, offset=None):
        # What the receiver decodes on a quiet link: the stair centroid
        # (soft decode) or the stair middle (plain geometric read).
        off = self._serpentine_zero if offset is None else offset
        if SOFT_DECODE:
            return sk.serpentine_stair_centroids(_f8(hosts), self.level, self.step,
                                                 off, host_factor)
        return sk.serpentine_stair_values(_f8(hosts), self.level, self.step, off)

    def encode(self, plane, index, leg=0, phase=None, guest_factor=1.0):
        if adaptive_guest_range and int(index) != host_level:
            raise ValueError(f'adaptive guest range fixes the host ladder at level {host_level}')
        hosts, flat, factor = self._normalised(plane, index, leg)
        offset = self._serpentine_zero if phase is None else self.offset(phase, leg)
        activity = self._activity(self._stairs(hosts, offset))
        scale = self._guest_scale(activity, leg)
        guests = sk.normalised_guests(flat, self.guest_position[leg], self.guest_sd[leg],
                                      self.guest_slot, scale, factor*guest_factor, self.slots)
        return sk.serpentine_encode_frame(_f8(hosts), guests, self.level, self.step,
                                          self.scale, self.compander, offset)

    def soft(self, values, sigma, density, offset=None):
        off = self._serpentine_zero if offset is None else offset
        if SOFT_DECODE and sigma is not None:
            return sk.serpentine_soft_decode_frame(
                _f8(values), self.level, self.step, self.scale, _f8(sigma),
                density, self.compander, off, host_factor)
        return decoder(_f8(values), self.level, self.step, self.scale,
                       self._serpentine_decoder_table, off)

    def soften(self, readings, phase=None):
        got = {}
        for key, (values, sigma) in readings.items():
            if key == 'sum':
                # The sum is an unscaled continuous stock host read, never an
                # invented guest-bearing serpentine symbol.
                hosts, _ = decoder(
                    _f8(values), self._serpentine_plain_level, self.step, self.scale,
                    self._serpentine_decoder_table, self._serpentine_zero)
                guests = np.zeros(self.slots)
            else:
                offset = self._serpentine_zero if phase is None else self.offset(phase, key)
                hosts, guests = self._soft(values, sigma, self.density, offset)
            got[key] = (hosts, guests, _f8(sigma))
        return got

    def decode_paired(self, got, factor, averaged=False):
        out_plane = np.zeros(self.luma)
        shared = (self.host_position, self.host_mean, self.host_sd, self.common,
                  self.differ, self.ride, self.detail_position, factor)
        if 'sum' in got:
            sk.place_single(out_plane, got['sum'][0], *shared, 0.0)
        elif len(got) == 2:
            # Base and detail weighted by what each leg's noise and stair
            # error left of them (the stock paired placement). The plain
            # algebraic split divides the two legs' independent stair errors
            # by the ride, which on small rides showed as bent lines.
            (left, _, sigma_left), (right, _, sigma_right) = got[0], got[1]
            sk.place_pair(out_plane, left, right, sigma_left, sigma_right,
                          self.level, self.step, *shared, bool(averaged))
        else:
            (leg, (hosts, _, _)), = got.items()
            sk.place_single(out_plane, hosts, *shared, 1.0 if leg == 0 else -1.0)
        for leg, (hosts, guests, _) in got.items():
            if leg in (0, 1):
                self._place_guests(out_plane, leg, hosts, guests, factor)
        return out_plane

    fold_class.__init__ = init
    fold_class._stairs = stairs
    fold_class.encode = encode
    fold_class._soft = soft
    fold_class.soften = soften
    fold_class._decode_paired = decode_paired

    if adaptive_guest_range:
        def choose_guest_factor(self, plane, index, phase=None):
            if int(index) != host_level:
                raise ValueError(f'adaptive guest range fixes the host ladder at level {host_level}')
            flat = _f8(plane).ravel()
            host_factor = nf.RATIO**int(index)
            peak = 0.0
            for leg in range(self.legs):
                hosts, _, _ = self._normalised(plane, index, leg)
                offset = self._serpentine_zero if phase is None else self.offset(phase, leg)
                activity = self._activity(self._stairs(hosts, offset))
                scales = self._guest_scale(activity, leg)
                found, status = sk.adaptive_guest_peak(
                    flat, self.guest_position[leg], self.guest_sd[leg], scales,
                    host_factor)
                if status:
                    raise ValueError('invalid/non-finite guest input while selecting adaptive factor')
                peak = max(peak, found)
            factor = float(sk.adaptive_guest_factor_from_peak(peak))
            if factor == 0.0:
                raise OverflowError('frame exceeds adaptive guest factor 16')
            return factor

        fold_class.choose_guest_factor = choose_guest_factor

        original_place_guests = fold_class._place_guests

        def place_guests(self, out_plane, leg, hosts, guests, factor):
            guest_factor = getattr(_ADAPTIVE, 'guest_factor', 1.0)
            return original_place_guests(self, out_plane, leg, hosts, guests,
                                         factor*guest_factor)

        fold_class._place_guests = place_guests

    original_decode = fold_class.decode

    def decode(self, readings, index, phase=None, averaged=False, soft=None,
               guest_factor=1.0):
        if adaptive_guest_range and int(index) != host_level:
            raise ValueError(f'adaptive guest range fixes the host ladder at level {host_level}')
        if not self.paired and len(readings) > 1 and 'sum' not in readings:
            raise ValueError('serpentine mono pilot does not support paired-host readings')
        if not adaptive_guest_range:
            return original_decode(self, readings, index, phase, averaged, soft)
        previous = getattr(_ADAPTIVE, 'guest_factor', 1.0)
        _ADAPTIVE.guest_factor = float(guest_factor)
        try:
            return original_decode(self, readings, index, phase, averaged, soft)
        finally:
            _ADAPTIVE.guest_factor = previous

    fold_class.decode = decode

    def choose_level(self, plane):
        del plane
        if fixed_level is not None:
            return int(fixed_level)
        raise RuntimeError('serpentine pilot requires the matched harness to pin a level')

    fold_class.choose_level = choose_level
    if isolated_path is not None:
        nf.register_fold_class(isolated_path, fold_class)
        registry = getattr(nf, '_isolated_serpentine_registry', {})
        registry[isolated_key] = configuration
        nf._isolated_serpentine_registry = registry
        return nf
    nf._serpentine_codec_installed = True
    nf._serpentine_codec_configuration = configuration
    nf.ADAPTIVE_GUEST_RANGE = adaptive_guest_range

    if metadata_path is None:
        return nf
    if metadata_path.exists():
        with metadata_path.open(encoding='utf-8') as stream:
            metadata = json.load(stream)
    else:
        metadata = {
            'mapping': MAPPING, 'version': VERSION,
            'required_at_both_ends': True,
            'wire_id': None,
            'automatic_level_selection': ('fixed level' if fixed_level is not None else 'disabled'),
            'fixed_level': fixed_level,
            'compatibility': 'wire-incompatible with legacy V7; exact same mapping required at both ends',
            'noise_claim': 'clean geometric inverse; no noise-robustness claim',
            'families': {},
        }
    if metadata.get('mapping') != MAPPING or metadata.get('version') != VERSION:
        raise ValueError(f'incompatible serpentine metadata already exists: {metadata_path}')
    metadata['automatic_level_selection'] = ('fixed level' if fixed_level is not None else 'disabled')
    metadata['fixed_level'] = fixed_level
    metadata['step_factor'] = step_factor
    metadata['guest_factor'] = guest_factor
    metadata['adaptive_guest_range'] = adaptive_guest_range
    metadata['pilot_semantics'] = ('guest factors [1,2,4,8,16] at rows [-3,-2,-1,0,1]; host level -1'
                                   if adaptive_guest_range else 'host ladder level')
    metadata['inverse'] = 'exact forward-table inversion' if exact_inverse else 'uniform inverse LUT'
    # Keep metadata for the other family when this hook is called again.
    metadata.setdefault('families', {})
    metadata.setdefault('supported_families', ['mono', 'slices'])
    with metadata_path.open('w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write('\n')
    return nf


def _f8(values):
    return np.ascontiguousarray(values, dtype=np.float64)
