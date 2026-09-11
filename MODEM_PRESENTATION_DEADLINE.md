# Full frame or deadline preview

Base: `ae5c591662ac0638798129295d0b3db245b522a9` on `modem`.
Keep your existing receive command, including `--receiver reference` and your
working device/channel selection. This applies to both GUI receivers.

- A valid shared-time presentation timestamp remains authoritative. The GUI
  shows the latest available reconstruction at that instant, then refinements.
- Without such a timestamp, a full reconstruction displays immediately. A
  partial waits until half the last completed packet's received duration from
  the current packet start. The first packet uses its own duration estimate.
- Duration is packet sample length multiplied by measured dilation, divided by
  the modem sample rate. It excludes inter-packet gaps and the idle tail.
- Packet start is estimated from received sample age at the first result. This
  is a processing-time mapping, not a capture-hardware timestamp; capture backlog
  can affect that estimate. No shared clock or chrony behavior is changed.
- At the deadline, the best currently available partial is released even if no
  more input arrives. Later refinements use the existing display-only overlay.
  With no usable result, the held picture stays up.
- A complete result with missing coefficients is still treated as a partial.
  A full coefficient count is not a guarantee of image fidelity.

Only one reconstruction per packet is queued. A full update replaces its older
partial, preventing that partial from reappearing later. Unscheduled input keeps
only the latest packet's pending reconstruction. There are no sleeps, extra
payload decodes, new acquisition operations, or changes to saved image values.

In a real-time stream, the full packet ordinarily arrives after its halfway
point, so this policy will still show a partial in that case. It prefers the full
picture when buffering or an authoritative later timestamp makes it available
before presentation.

Seventeen focused tests passed with RuntimeWarning treated as errors:

```bash
python -W error::RuntimeWarning -m unittest modem_tests.test_presentation_deadline modem_tests.test_progressive_receive modem_tests.test_preview_overlay -v
```

The deadline tests use explicit timestamps, with no sleeps or real-time simulation.
No full suite, hardware benchmark, or live GUI trial was performed.
