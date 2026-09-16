# Scope web page and sound-output switching

`/scope` mirrors the main monitor page and adds a sound-output picker at the
bottom, defaulting to the system default output.

---

## 0. About the abandoned draft

Most of the backend was already in the working tree from the run that was cut
off at your session limit. I did not adopt it as-is — unfinished and untested is
the category most likely to contain the errors you asked me to remove. I
reviewed all of it. Results:

| flagged | verdict |
|---|---|
| `do_POST` docstring claims `/ascii/size` is dead | **True.** `grep -rn "ascii/size"` finds two `fetch()` calls in `lightweight_monitor.py` and no handler; `def get_ascii_dimensions` is defined nowhere. See §4. |
| `with scope:` replaced by `if True:` — teardown lost? | **Unfounded.** Teardown is present in a `finally:`. The stated reason for dropping the with-block is correct: the device swap rebinds `scope`, so `__exit__` would fire on the already-closed old object. |
| `_swap_device` reopens under a live `BufferedSource` | **Real, and fixed.** See §3. |

Plus two more I found on review, and one piece of leftover scaffolding. All
fixed below.

`templates/scope.html` from that run is at `/tmp/quarantine/scope.html.unverified`
if you want it. I did not use it — I rewrote the page against the actual
`HTML_TEMPLATE`, with one substantive change (§2).

---

## 1. What you get

`GET /scope` — verified serving at 10852 bytes. No route was added for it:
`_serve_dynamic_template()` already maps `/scope` to `templates/scope.html`,
with traversal guarded and the `.html` extension enforced.

Three sections, styled from `HTML_TEMPLATE` so it reads as the same site:

- **Scope** — the sixteen `scope_*` fields, in a fixed order rather than
  whatever order `/data` happens to serialise. `scope_underruns` turns red
  above zero; on a Pi it is the number that tells you the generator is not
  keeping up.
- **Playback & system** — everything non-`scope_` from `/data`, the same set
  the main page shows.
- **Sound output** — the picker, deliberately last. It is the only control on
  the page that interrupts audio, so it should not sit above the numbers you
  would use to decide whether to touch it.

The picker's first option is **System default output**, pre-selected, showing
the default device's name beside it. That matches what the process does with no
`--device`. A **Use system default** button returns to it in one click.

Options send the device **name**, never the index — PortAudio indices reshuffle
on replug, so an index chosen when the page loaded can point somewhere else by
the time you click. This is the same reasoning as the `SCOPE_DEVICE` comment in
`settings.py`.

If the selected device runs at a different sample rate, an amber line appears
before you commit: `samples/trace = samplerate/fps`, so a different rate is a
different sample budget and therefore a different grid. Switching recalibrates.

---

## 2. One change I made to the draft's page

The draft built rows with `innerHTML` and string concatenation. Device names
come from the operating system, and on Linux they routinely contain characters
from the ALSA description string. That is untrusted input rendered as markup.

The rewrite uses `document.createElement` and `textContent` throughout. Same
appearance, no injection path. Worth doing here specifically because you are
adding a mutating endpoint to a page that previously only displayed numbers.

---

## 3. Fixes to the backend

### 3a. Realtime cannot survive a device swap — now refused

`BufferedSource` wraps a generator built around the current
`samples_per_frame`. A different device can have a different default sample
rate, which changes that — and the generator would go on emitting the old
length. The audio would keep playing and the picture would be quietly wrong.

Realtime mode now refuses the swap with a message telling you to restart with
`--device` instead. Refusing is honest; swapping would look like it worked.

### 3b. Stale monitor fields after a swap

The draft updated `scope_device`, `scope_samplerate`, `scope_samples_per_trace`
and `scope_grid` after a swap, but left `scope_refresh_hz`, `scope_picture_hz`
and `scope_samples_per_cell` untouched. Since a new device can mean a new
sample rate, the page would report the **old** refresh rate against the **new**
device — worse than reporting nothing. All three now update.

### 3c. Leftover scaffolding

The abandoned edit had wrapped the main loop in `if True:` to avoid
re-indenting 205 lines. Valid Python, but it left the block at a 2-space step
in a 4-space file. Removed, and the loop and its `finally:` normalised back to
4-space indentation. Purely cosmetic; verified by compile.

### What I kept, and why

`_swap_device` closes the old stream **before** opening the new one. That is
correct, and the draft's reasoning holds: exclusive-mode routes (raw ALSA
`hw:`, WASAPI exclusive) refuse a second handle, so holding both would fail on
exactly the devices worth using.

It also resets the sweep chain rather than carrying `sweep["end"]` across.
Correct — after a teardown the beam is not where the chain says it is, and
continuing would put a full-screen jump in the first trace on the new device.

And it recalibrates rather than reusing the old grid, for the reason your
handoff already gives: the grid depends on the sample budget, which is a fact
about the device.

---

## 4. A finding about your main site, unrelated to scope

**The ASCII dimension panel on the main monitor page has never worked.**

`HTML_TEMPLATE` in `lightweight_monitor.py` fetches `/ascii/size` at lines 197
and 209 (GET and POST). There is no handler for that path anywhere —
`web_service.py` had no `do_POST` at all before this change, and `do_GET` has
no `/ascii/size` branch. `lightweight_monitor.py` also does
`from shared_state import get_ascii_dimensions` inside a `try/except
ImportError`, and `shared_state.py` does not define it — so the import silently
fails and `_ascii_dims_available` stays False.

The panel renders, accepts input, and does nothing. I have not touched it; the
new `do_POST` handles `/scope/device` only. Flagging it because you asked for
hallucination-induced errors removed, and a control panel that has never been
wired to anything is one, just an older one.

---

## 5. Tested

Real handler, real port, stubbed audio:

```
GET /scope        -> 10852 bytes | title ok: True | picker present: True
GET /data         -> 58 keys | 2 devices | default is ['MacBook Pro Speakers']
POST named device    -> {'ok': True, 'requested': 'BlackHole 2ch'} | parked: pending=True
POST system default  -> {'ok': True, 'requested': None}            | parked: pending=True
POST oversize     -> rejected 413 (good)
```

`""` and `null` both resolve to `None`, which is the system default — that is
what the page's first option sends.

---

## 6. Files

| file | status |
|---|---|
| `templates/scope.html` | **new** |
| `web_service.py` | **new to your set** — `do_POST` for `/scope/device`, +38 lines |
| `scope_display.py` | `request_device`, `device_status`, `_swap_device`, `scope_devices`, plus §3 fixes |
| `main.py`, `settings.py`, `scope_bake.py`, `scope_out.py`, `test_scope_pair.py`, `scope_tap.py`, `scope_sidecar.py` | unchanged this pass |

`web_service.py` is shared with all four other modes. The change is additive —
a `do_POST` where none existed — so no existing behaviour moves.

Everything compiles. Interlace (45 refs), orientation (4 `invert_y=False`) and
mix (3 `mix_field_i`) all intact.

---

## 7. One thing to settle before this faces anything

`run_monitor_server()` binds `WEB_HOST`, default `127.0.0.1`. Until now the
monitor was read-only, so that default was a formality. `/scope/device` is a
mutating endpoint, and `setup_nginx.sh` is in the repo — so it is worth
deciding deliberately whether it is reached over an SSH tunnel, behind nginx
with auth, or on a trusted LAN. There is no authentication on it. That is
consistent with the rest of the monitor, which is the argument for leaving it
and the argument for not exposing it.

Still open from earlier passes: no `vi-scope.service`, no `--scope` shortcut in
`run_app.sh`, and the PipeWire-vs-ALSA question that decides the unit's shape.