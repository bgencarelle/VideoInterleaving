"""Regression checks for the no-Z stochastic scope renderer."""
import numpy as np

from scope_bake import (StochasticEmitter, TraceEmitter, composite_luma,
                        stochastic_luma)
from scope_controls import KeyMap, as_flags


def _portrait():
    lum = np.zeros((96, 72), dtype=np.float64)
    y, x = np.ogrid[:96, :72]
    lum[((x - 36) / 24) ** 2 + ((y - 48) / 38) ** 2 < 1] = 0.35
    lum[((x - 36) / 9) ** 2 + ((y - 48) / 16) ** 2 < 1] = 1.0
    return lum


def test_stochastic_shape_range_and_determinism():
    lum = _portrait()
    a = stochastic_luma(lum, 1600, rng=np.random.default_rng(7))
    b = stochastic_luma(lum, 1600, rng=np.random.default_rng(7))
    assert a.shape == (1600, 2)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()
    assert np.abs(a).max() <= 0.900001
    assert np.array_equal(a, b)


def test_stochastic_trace_chaining():
    emitter = StochasticEmitter(48000, 1600, seed=3)
    first = emitter.emit(_portrait())
    second = emitter.emit(_portrait())
    # A trace boundary is only an audio-buffer boundary. The next target is
    # chosen on the next DAC interval instead of duplicating the old endpoint
    # and losing one target decision per trace.
    assert not np.array_equal(first[-1], second[0])
    assert emitter._count == 3200


def test_brightness_is_visit_density():
    lum = np.full((96, 72), 0.25, dtype=np.float64)
    lum[34:62, 26:46] = 1.0
    frame = stochastic_luma(
        lum, 1600, rng=np.random.default_rng(4), edge_gain=0.0)
    # The bright rectangle is only 8.1% of the source, so it should be enriched
    # without erasing the surrounding 25% midtones. Gamma 6 put >50% of the
    # beam here and reduced portrait skin to disconnected highlight islands.
    in_bright = (np.abs(frame[:, 0]) < 0.19) & (np.abs(frame[:, 1]) < 0.18)
    assert 0.20 < in_bright.mean() < 0.45


def test_blank_luminance_returns_none():
    assert stochastic_luma(np.zeros((20, 20)), 200) is None


def test_stochastic_selects_raw_baked_luminance_and_legacy_still_works():
    class Lib:
        def __init__(self, channels):
            self.value = np.zeros((4, 3, channels), dtype=np.uint8)
            self.value[..., 0] = 25       # raster-preconditioned luminance
            self.value[..., 1] = 255
            if channels >= 3:
                self.value[..., 2] = 200  # raw stochastic luminance

        def __len__(self):
            return 1

        def thumb(self, _index):
            return self.value

    assert np.allclose(composite_luma(Lib(3), 0, None, 0), 25 / 255)
    assert np.allclose(composite_luma(Lib(3), 0, None, 0, raw=True), 200 / 255)
    assert np.allclose(composite_luma(Lib(2), 0, None, 0, raw=True), 25 / 255)


def test_single_pixel_never_parks_the_beam():
    emitter = StochasticEmitter(48000, 200, seed=1)
    first = emitter.emit(np.ones((1, 1)))
    second = emitter.emit(np.ones((1, 1)))
    for frame in (first, second):
        assert len(np.unique(frame, axis=0)) > 100
        assert np.hypot(*np.ptp(frame, axis=0)) > 0.0
    assert np.linalg.norm(first[-1] - second[0]) < 0.04


def test_walk_clock_is_not_a_high_rate_dac_clock():
    lum = _portrait()
    emitter = StochasticEmitter(96000, 3200, walk_hz=48000, seed=11)
    frame = emitter.emit(lum)
    # The same 1/30 second contains 1,600 target decisions at 48 and 96 kHz;
    # the latter merely samples each target interval twice.
    assert emitter._count == 1600
    changes = np.count_nonzero(np.any(np.diff(frame, axis=0), axis=1))
    assert changes < 1700
    assert changes > 1000


def test_auto_stride_scales_osci_source_pixels_to_bake_width():
    emitter = StochasticEmitter(48000, 200, stride=0)
    for width, expected in ((96, 1), (128, 1), (256, 2), (480, 4)):
        assert emitter._stride_for_width(width) == expected


def test_image_update_does_not_reset_continuous_walk():
    emitter = StochasticEmitter(48000, 800, seed=5)
    emitter.emit(_portrait())
    count = emitter._count
    changed = _portrait()
    changed[:, :20] = 0.8
    emitter.emit(changed)
    assert count == 800
    assert emitter._count == 1600


def test_live_mode_cycles_all_three_renderers():
    state = {"mode": "vector", "raster": False, "gamma": 2.2,
             "raster_gamma": 2.2, "stochastic_gamma": 2.0}
    keys = KeyMap(state)
    keys.feed("v")
    assert state["mode"] == "raster"
    assert state["gamma"] == 2.2
    keys.feed("v")
    assert state["mode"] == "stochastic"
    assert state["gamma"] == 2.0
    assert "--scope-mode stochastic" in as_flags(state)
    assert "--scope-gamma 2" in as_flags(state)
    keys.feed("v")
    assert state["mode"] == "vector"
    assert state["gamma"] == 2.2


def test_live_mode_cycle_respects_fixed_scheduler():
    state = {"mode": "raster", "raster": True, "mode_locked": True}
    keys = KeyMap(state)
    keys.feed("v")
    assert state["mode"] == "raster"
    assert "unavailable" in keys.message


def test_runtime_mode_handoff_starts_at_actual_beam_endpoint():
    import scope_display

    class Lib:
        thumbs = True

        def __len__(self):
            return 1

        def thumb(self, _index):
            thumb = np.zeros((48, 36, 2), dtype=np.uint8)
            thumb[8:40, 8:28, 0] = 210
            thumb[8:40, 8:28, 1] = 255
            return thumb

        def frame(self, _index):
            square = np.array([
                [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                [-0.5, 0.5], [-0.5, -0.5]], dtype=np.float32)
            return [square], [1]

    class FakeScope:
        samplerate = 48000
        samples_per_frame = 800

        def __init__(self):
            self.frames = []

        def show_frame(self, frame):
            self.frames.append(frame)

    scope_display.Scope._tap_until = 0
    scope = FakeScope()
    lib = Lib()
    raster = TraceEmitter(48000, 800, dc_comp=30)
    stochastic = StochasticEmitter(48000, 800, seed=9, dc_comp=30)
    common = (scope, lib, None, 0)
    vector_end = scope_display._emit(
        *common, "vector", {}, "alternate", 2.2, 0.02, 1.0, None, 0.02,
        lowpass=3000, emitter=raster, stochastic_emitter=stochastic)
    stochastic_end = scope_display._emit(
        *common, "stochastic", {}, "alternate", 2.2, 0.02, 1.0, None, 0.02,
        lowpass=3000, emitter=raster, stochastic_emitter=stochastic,
        beam_start=vector_end)
    assert np.array_equal(scope.frames[-1][0], vector_end)
    continuous_end = scope_display._emit(
        *common, "stochastic", {}, "alternate", 2.2, 0.02, 1.0, None, 0.02,
        lowpass=3000, emitter=raster, stochastic_emitter=stochastic,
        beam_start=stochastic_end)
    assert not np.array_equal(scope.frames[-1][0], stochastic_end)
    scope_display._emit(
        *common, "raster", {}, "alternate", 2.2, 0.02, 1.0, None, 0.02,
        lowpass=3000, emitter=raster, stochastic_emitter=stochastic,
        beam_start=continuous_end)
    assert np.array_equal(scope.frames[-1][0], continuous_end)
