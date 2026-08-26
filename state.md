# Where we are

You pushed once mid-session, at commit `a166b2c`. Everything since then is
staged in `scope_files/` and not yet in the repo.

---

## Install this

15 files. Nine replace repo-root files, three are new, one goes in
`templates/`, one in `utilities/`.

```
scope_files/main.py                 -> main.py
scope_files/settings.py             -> settings.py
scope_files/scope_bake.py           -> scope_bake.py
scope_files/scope_display.py        -> scope_display.py
scope_files/scope_out.py            -> scope_out.py
scope_files/scope_screen.py         -> scope_screen.py
scope_files/scope_tap.py            -> scope_tap.py
scope_files/scope_sidecar.py        -> scope_sidecar.py
scope_files/test_scope_pair.py      -> test_scope_pair.py
scope_files/web_service.py          -> web_service.py
scope_files/utilities/convert_to_xy.py -> utilities/convert_to_xy.py
scope_files/templates/scope.html    -> templates/scope.html          NEW
scope_files/test_scope_parity.py    -> test_scope_parity.py          NEW
scope_files/bake_advisor.py         -> bake_advisor.py               NEW
scope_files/scope_profile.py        -> scope_profile.py    (unmodified, yours)
```

Then:

```bash
rm -rf __pycache__
python verify_scope_files.py      # expect only xy_display.py MISSING
python test_scope_parity.py       # expect PARITY OK
```

**No rebake.** Nothing this session changed the baked format.

Two things to clean in the repo while you're there: `scope_website.md.py` is a
markdown doc with a `.py` extension (fails `compileall`), and
`verify_scope_files.py` has a phantom `xy_display.py` entry that displaced the
`scope_profile.py` check.

---

## Run it

```bash
python main.py --mode scope --xy-dir images_xy --scope-raster \
       --scope-trim 0.10 --scope-fields 2 --device BlackHole
```

Then `http://127.0.0.1:8890/scope`.

**Check the banner for your sample rate.** If it says 48000, that is 47 rows
instead of 66 — the single biggest number in the whole system, and it is a
device setting, not a code change.

---

## What actually got fixed (bugs, not tuning)

| bug | effect |
|---|---|
| Interlace grid sized per trace, not per picture | interlace was *worse* than progressive at the same rate |
| Raster emitted per index, not per trace | refresh pinned to 30 Hz; flyback on repeated frames |
| `sweep["end"]` advanced on dropped frames | silent chain break |
| Vector double-flipped Y | vector mode upside down, and it depended on `--scope-lowpass` |
| Mix sized its grid per trace | raster half of mix at 0.24x the cells, half of it avoidable |
| `scope_screen` re-derived its grid every frame | the per-frame-adaptation flicker your handoff warns about |
| Scope reserved a port it never binds | refused to start alongside web; could trip systemd's StartLimit |
| Preview tap captured one *field* | `/scope` showed 50% of scanlines at fields=2, **26% at fields=4** |

The last one matters most given the website is the display for most people.

## What got added

- `--scope-fields N` — interlace
- `--scope-border F` — fixed rectangle pinning the framing
- `--scope-row-bias F` — trade columns for rows (try 1.3–1.5)
- `--scope-dc-comp HZ` — the flag both handoffs documented but never existed
- `/scope` web page with live MJPEG preview and a sound-output picker
- `TraceEmitter` — one implementation of "luminance to trace", shared by both drivers
- `test_scope_parity.py` — fails if the two paths diverge again
- `bake_advisor.py` — measures what `--thumb-width` your content needs

---

## The three things worth knowing

**1. Rows are the cheap axis, columns are the expensive one.** Vertical detail
is row position — a ~2 kHz signal that passes any output filter. Horizontal
detail is dwell modulation inside a sweep, sitting at 25–50 kHz, past a typical
22 kHz reconstruction filter. Measured MTF: vertical ~4x horizontal at 8
cycles. This is why `--scope-row-bias` works and why extra *columns* from a
higher sample rate are wasted while extra *rows* are not.

**2. The website is the display for most users.** `scope_bake.preview_frame` is
the primary renderer, not a diagnostic. Anything that improves the sample
stream helps both paths; anything that only affects how an analog tube
integrates (sweep mode, persistence) helps only the hardware minority and
should not compromise the web view.

**3. 48 kHz is the honest design baseline**, not 96. At 48 kHz the real
composite gets 35x47; at 96 kHz, 50x66.

---

## Open, in the order I would take them

1. **Confirm your sample rate** from the banner. Five seconds, biggest number.
2. **Try `--scope-row-bias 1.5`** and judge in motion. Measured better on your
   own face; one frame, one pose, so it needs your eyes.
3. **`_walk`'s horizontal smearing.** The renderer delivers ~40% of its own
   column-limited Nyquist and I do not know why. Largest unexplained number
   left, and the only remaining horizontal lever.
4. **Border connector diagonal.** Visible across the picture on full-frame
   composites. My "17 samples, negligible" measurement used matted content that
   flattered it.
5. **`vi-scope.service`.** Blocked on whether the target runs PipeWire (user
   service) or bare ALSA (system service, needs `SupplementaryGroups=audio`).
6. **Bake the tone-mapping levels** — your handoff's open item #2, still open.
7. **Prerender queue.** 84–94% CPU idle; one dropped trace currently costs a
   field. Insurance, only worth it if you actually see blinks.

---

## Things I got wrong and corrected

Listed because you asked me to strip hallucination-induced errors, and I
introduced some of my own.

- Claimed autofit overshoots to 0.25 samples/cell. Wrong denominator — it hits
  1.12 against a 1.0 target and is correct.
- Recommended dropping to `--thumb-width 96`. Wrong: it clips at a ~30%
  subject.
- Said `precompensate_hpf` was dead code. It is called by `scope_screen.py:439`.
- Said `--scope-ask`/`--scope-device` do not exist. They do.
- Recommended 192 kHz, then over-retracted it. Extra columns are wasted; extra
  *rows* are real. You were right about the filter, I was wrong to drop the
  whole idea.
- Said interlace does not reduce flicker. It does — by staggering decay phases
  across rows, 89% -> 42% area swing. It does not slow any row's decay, which
  is a different claim.
- Tested three perceptual hacks on synthetic content and reported the row-split
  conclusion before you pointed out the test data was unrepresentative. All
  three came out negative on real content.