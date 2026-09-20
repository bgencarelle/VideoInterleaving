#!/usr/bin/env python3
"""Compare V6 placements under a synthetic shared tape-lift (spacing-loss) model.

TEMPORARY DIAGNOSTIC. This is not a model of any particular deck; real tape
captures remain authoritative. It exists because the older synthetic dropout
(silence on both tracks) never exercises the common tape failure: a short,
partial, frequency-dependent fade shared by both tracks.

Model, applied at the 48 kHz reference rate:

* Head-to-tape lift d(t) in micrometres, shared by both tracks, made of seeded
  raised-cosine events on a fixed ABSOLUTE timeline (identical for every
  variant, whatever its frame duration). Each track sees d(t)*(1+e) with a
  small per-event, per-track difference e.
* Wallace spacing loss: attenuation(f, t) = 54.6 * d(t) / lambda dB, with
  lambda = v/f at v = 4.7625 cm/s (1 7/8 ips). Applied by short-time Fourier
  weighting (256-sample window, 64-sample hop).
* Optional track skew: a slowly varying right-channel delay (microseconds),
  plus slow per-track gain wander.
* Treble-weighted hiss, as in tools/test_tape_matrix.py.

Scores are against each variant's own clean decode, per frame, split by plane,
plus coefficient-domain error of the 720-value foundation versus the detail.
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import butter, istft, sosfilt, stft

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from animation_modem import transport3 as V3                             # noqa: E402
from animation_modem import v6                                           # noqa: E402
from animation_modem.imaging import image_values, prepare_image           # noqa: E402

RATE = V3.REFERENCE_RATE
TAPE_SPEED_UM_S = 47625.0          # 1 7/8 ips
PLANES = (('Y', slice(0, 96*80)), ('Cb', slice(96*80, 96*80+48*40)),
          ('Cr', slice(96*80+48*40, None)))


@dataclass(frozen=True)
class LiftCase:
    name: str
    rate_per_s: float = 0          # lift events per second
    peak_um: tuple = (0, 0)        # log-uniform event peak range
    duration_ms: tuple = (4, 40)   # uniform event duration range
    base_um: float = 0             # steady lift (worn/poor contact)
    track_diff: float = .15        # per-event, per-track lift difference
    skew_us: float = 0             # peak slow right-channel delay wander
    gain_wander_db: float = 0      # peak slow per-track gain wander
    hiss_dbfs: float | None = None
    mute: int | None = None        # legacy control: silence one track


CASES = (
    LiftCase('clean'),
    LiftCase('hiss-45', hiss_dbfs=-45),
    LiftCase('lift-mild', rate_per_s=3, peak_um=(.3, 1.5), hiss_dbfs=-45),
    LiftCase('lift-severe', rate_per_s=4, peak_um=(1, 4), hiss_dbfs=-45),
    LiftCase('lift-worn', rate_per_s=3, peak_um=(.3, 1.5), base_um=.6,
             hiss_dbfs=-45),
    LiftCase('lift-skew', rate_per_s=4, peak_um=(1, 4), track_diff=.35,
             skew_us=8, gain_wander_db=1, hiss_dbfs=-45),
    LiftCase('mute-left', mute=0),
    LiftCase('mute-right', mute=1),
)


def lift_events(case, seconds, seed):
    """Seeded (start_s, duration_s, peak_um, left_factor, right_factor)."""
    if not case.rate_per_s:
        return []
    rng = np.random.default_rng(seed)
    events, t = [], 0.0
    while True:
        t += rng.exponential(1/case.rate_per_s)
        if t >= seconds:
            return events
        lo, hi = case.peak_um
        peak = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        duration = float(rng.uniform(*case.duration_ms))/1000
        diff = rng.uniform(-case.track_diff, case.track_diff, 2)
        events.append((t, duration, peak, 1+float(diff[0]), 1+float(diff[1])))


def lift_profile(case, events, times):
    lift = np.full((len(times), 2), case.base_um, float)
    for start, duration, peak, left, right in events:
        inside = (times >= start) & (times < start+duration)
        shape = .5-.5*np.cos(2*np.pi*(times[inside]-start)/duration)
        lift[inside, 0] += peak*left*shape
        lift[inside, 1] += peak*right*shape
    return lift


def spacing_loss(audio, case, events):
    if not events and not case.base_um:
        return audio
    nperseg, hop = 256, 64
    out = np.empty_like(audio)
    for channel in range(2):
        f, t, z = stft(audio[:, channel], fs=RATE, window='hann',
                       nperseg=nperseg, noverlap=nperseg-hop, boundary='even')
        lift = lift_profile(case, events, t)[:, channel]
        loss_db = -54.6*lift[None, :]*f[:, None]/TAPE_SPEED_UM_S
        _, y = istft(z*10**(loss_db/20), fs=RATE, window='hann',
                     nperseg=nperseg, noverlap=nperseg-hop, boundary=True)
        out[:, channel] = y[:len(audio)]
    return out


def skew_and_wander(audio, case):
    t = np.arange(len(audio))/RATE
    out = audio.copy()
    if case.skew_us:
        delay = case.skew_us*1e-6*RATE*np.sin(2*np.pi*.3*t + .7)
        base = np.arange(len(audio), dtype=float)
        out[:, 1] = np.interp(base-delay, base, audio[:, 1])
    if case.gain_wander_db:
        for channel, phase in ((0, .3), (1, 2.1)):
            db = case.gain_wander_db*np.sin(2*np.pi*.2*t + phase)
            out[:, channel] *= 10**(db/20)
    return out


def tape_hiss(count, dbfs, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((count, 2))
    high = sosfilt(butter(2, 1800, btype='highpass', fs=RATE, output='sos'),
                   noise, axis=0)
    high /= max(np.sqrt(np.mean(high*high)), 1e-12)
    return high*10**(dbfs/20)


def impair(audio, case, events, seed):
    x = np.asarray(audio, float)
    if case.mute is not None:
        x = x.copy()
        x[:, case.mute] = 0
        return x.astype(np.float32)
    x = skew_and_wander(spacing_loss(x, case, events), case)
    if case.hiss_dbfs is not None:
        x = x + tape_hiss(len(x), case.hiss_dbfs, seed)
    return np.clip(x, -1, 1).astype(np.float32)


def decode(audio, layout, coder):
    receiver = V3.Receiver(layout, coder, pulse_only=True, fast=True,
                           coders={0: coder},
                           candidates=[(layout, coder, {0: coder})])
    out = []
    for start in range(0, len(audio), 1024):
        out.extend(receiver.feed(audio[start:start+1024]))
    return out + receiver.flush()


def coefficients(coder, values):
    return coder.base.forward(values)/coder.base.gains


def variants(transform):
    return (
        ('v6-foundation', V3.WIRE_V6, v6.coder_for(transform)),
        ('v6-tape', V3.WIRE_V6, v6.tape_coder(transform)),
        ('nocopy-middle', V3.WIRE_V6, v6.nocopy_coder(transform)),
        ('nocopy-tape', V3.WIRE_V6, v6.tape_coder(transform, False)),
        ('full-repeat', V3.WIRE_V6_REPEAT, v6.full_repeat_coder(transform)),
    )


def source_image(path):
    if path is not None:
        image = Image.open(path).convert('RGB')
    else:
        # Natural-image stand-in when the face source is not in the checkout.
        from scipy.datasets import face
        image = Image.fromarray(face())
    width, height = image.size
    crop = min(width, height*3//4)
    left = (width-crop)//2
    return image.crop((left, 0, left+crop, crop*4//3))


def run(args):
    prepared = prepare_image(source_image(args.source), 'auto', 'lanczos')
    foundation = {'dct': v6._dct_foundation(), 'wavelet': v6._wavelet_foundation()}
    rows = []
    for transform in args.transforms:
        for label, layout, coder in variants(transform):
            values = image_values(prepared, coder.grids)
            frames = int(np.ceil(args.seconds*layout.fps))
            audio = np.concatenate([
                V3.encode(values, layout, coder, i+1, i+1, frames, profile=0)
                for i in range(frames)])
            clean = {r.absolute: r.values for r in decode(audio, layout, coder)
                     if r.identity == 'verified_header' and r.values is not None}
            reference = next(iter(clean.values()))
            clean_coef = {k: coefficients(coder, v) for k, v in clean.items()}
            source_rmse = round(float(np.mean([np.sqrt(np.mean((v-values)**2))
                                               for v in clean.values()])), 4)
            detail = np.setdiff1d(np.arange(v6.ORIGINAL_VALUES), foundation[transform])
            for index, case in enumerate(CASES):
                if args.only and case.name not in args.only:
                    continue
                events = lift_events(case, args.seconds + 2, seed=100+index)
                started = time.perf_counter()
                results = decode(impair(audio, case, events, 200+index), layout, coder)
                elapsed = time.perf_counter()-started
                pictures = [r for r in results if r.values is not None]
                verified = sum(r.identity == 'verified_header' for r in results)
                per_plane = {name: [] for name, _ in PLANES}
                total, coef_f, coef_d = [], [], []
                for r in pictures:
                    ref = clean.get(r.absolute, reference)
                    err = r.values-ref
                    total.append(float(np.sqrt(np.mean(err**2))))
                    for name, part in PLANES:
                        per_plane[name].append(float(np.sqrt(np.mean(err[part]**2))))
                    ref_coef = clean_coef.get(r.absolute, clean_coef[
                        next(iter(clean_coef))])
                    ce = coefficients(coder, r.values)-ref_coef
                    coef_f.append(float(np.sqrt(np.mean(ce[foundation[transform]]**2)) /
                                        np.sqrt(np.mean(ref_coef[foundation[transform]]**2))))
                    coef_d.append(float(np.sqrt(np.mean(ce[detail]**2)) /
                                        np.sqrt(np.mean(ref_coef[detail]**2))))
                mean = lambda x: round(float(np.mean(x)), 4) if x else None
                p90 = lambda x: round(float(np.percentile(x, 90)), 4) if x else None
                row = {'transform': transform, 'variant': label, 'case': case.name,
                       'fps': round(layout.fps, 2), 'frames': frames,
                       'clean_vs_source': source_rmse,
                       'verified': verified, 'pictures': len(pictures),
                       'held': frames-len(pictures),
                       'rmse': mean(total), 'rmse_p90': p90(total),
                       'Y': mean(per_plane['Y']), 'Cb': mean(per_plane['Cb']),
                       'Cr': mean(per_plane['Cr']),
                       'chroma_p90': p90([max(a, b) for a, b in
                                          zip(per_plane['Cb'], per_plane['Cr'])]),
                       'found_err': mean(coef_f), 'detail_err': mean(coef_d),
                       'damaged_frames': int(sum(e > .05 for e in total)),
                       'lift_events': len([e for e in events if e[0] < args.seconds]),
                       'decode_s_per_frame': round(elapsed/max(frames, 1), 4)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, default=None,
                        help='RGB source (default: scipy face stand-in)')
    parser.add_argument('--seconds', type=float, default=6.0)
    parser.add_argument('--transforms', nargs='+', default=['dct', 'wavelet'],
                        choices=['dct', 'wavelet'])
    parser.add_argument('--only', action='append', default=[])
    parser.add_argument('--json', type=Path, default=None)
    args = parser.parse_args(argv)
    rows = run(args)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
