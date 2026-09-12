# Native waveform receiver

The `waveform` receiver treats ADC samples as the authoritative band-limited
waveform. It bypasses the continuous capture resampler. Each packet measures
its own sample coordinate scale from the known preamble/training references,
then uses fractional interpolation only for the modem symbol windows.

The device's reported sample rate is used to open the stream and express
diagnostic/presentation durations in seconds. It is not used to steer the modem
clock. Native capture reads 256 samples per block. Frame gaps have no
timing meaning. The existing progressive receiver remains the default.

```bash
python utilities/modem_v2_check.py live-receive --receiver waveform
```

This is an experimental native-waveform implementation. It reuses the existing
packet format and progressive symbol consumer, including its equalization.
Timing errors can still affect image quality. It does not add the future V3
multi-tone reference markers or independent pitch estimator. A sample rate
whose Nyquist limit is below the highest transmitted carrier cannot recover
that carrier; no interpolation can restore aliased information.

The portable unit tests generate their own packets and require all frame IDs,
coefficient RMS error below .003 at 48 kHz and simulated 96 kHz, one FFT per
symbol, silence recovery, and mocked native live startup without a resampler.
Run them with:

```bash
python -W error::RuntimeWarning -m unittest modem_tests.test_waveform_receiver modem_tests.test_progressive_receive modem_tests.test_capture -v
```

A separate short check of the 16 supplied converted faces recovered all frames
at both rates and at gains 1 and .01. Maximum coefficient RMS error was .000122
at 48 kHz and .000594 at simulated 96 kHz. These are synthetic checks, not an
analog loopback validation or performance guarantee.

Known limitations: detection requires distinguishable L-only preamble and
R-only training events. Severe crosstalk, mono input, independent pitch shifts,
and large channel delay differences are not solved. This receiver emits a
completed packet reconstruction only; it does not publish mid-packet previews
or implement the V3 dirty-packet fallback policy. Existing display hold behavior
and the shared-time scheduler are unchanged. No local correlation candidate
search or whole-payload decode retry is used.

This complete patch is based on modem commit
0f4d8903f5ff21cb5327745af3857ac4554bfab7 and requires no earlier patches.
