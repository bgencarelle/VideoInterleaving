"""Regression checks for the no-Z stochastic and fusion scope renderers."""
import tempfile
from pathlib import Path

import numpy as np

from scope_bake import (StochasticEmitter, TraceEmitter, TriangleMixScheduler,
                        calibrate, composite_luma, stochastic_luma,
                        fuse_density, fusion_probability,
                        normalize_fusion_components, vector_density)
from utilities.convert_to_xy import THUMB_W
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


def test_mismatched_thumbnail_geometry_is_rejected_not_half_composited():
    class Lib:
        def __init__(self, shape):
            self.value = np.zeros(shape + (3,), dtype=np.uint8)
            self.value[..., 1] = 255

        def __len__(self):
            return 1

        def thumb(self, _index):
            return self.value

    try:
        composite_luma(Lib((128, 96)), 0, Lib((341, 256)), 0, raw=True)
    except ValueError as exc:
        assert "geometry differs" in str(exc)
    else:
        raise AssertionError("mismatched main/float bakes were silently mixed")


def test_fixed_calibration_respects_autofit_toggle():
    class Lib:
        def __init__(self):
            self.thumbs = np.zeros((4, 100, 100, 3), dtype=np.uint8)
            self.thumbs[..., 1] = 255
            self.thumbs[:, 35:65, 35:65, 0] = 220

        def thumb(self, index):
            return self.thumbs[index]

    fitted = calibrate([Lib()], [], 1600, trim=0.1, autofit=True)
    full = calibrate([Lib()], [], 1600, trim=0.1, autofit=False)
    assert fitted["grid_rows"] > full["grid_rows"]
    assert fitted["grid_cols"] > full["grid_cols"]


def test_device_swap_recalibration_keeps_row_bias_and_autofit():
    import scope_display
    import scope_out

    class Stream:
        device = "fake"

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    class FakeScope:
        samplerate = 48000
        samples_per_frame = 800

        def __init__(self, **_kwargs):
            self.stream = Stream()

    old = FakeScope()
    seen = {}
    original_scope = scope_out.Scope
    original_resolve = scope_out.resolve_device
    original_calibrate = scope_display.calibrate
    try:
        scope_out.Scope = FakeScope
        scope_out.resolve_device = lambda spec: spec

        def fake_calibrate(*_args, **kwargs):
            seen.update(kwargs)
            return {}

        scope_display.calibrate = fake_calibrate
        scope_display._swap_device(
            old, "fake", None, 60, None, [], [], 1.0, 0.1, None, 2,
            1.5, False)
    finally:
        scope_out.Scope = original_scope
        scope_out.resolve_device = original_resolve
        scope_display.calibrate = original_calibrate

    assert seen["row_bias"] == 1.5
    assert seen["autofit"] is False


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


def test_thumbnail_default_retains_stochastic_spatial_detail():
    assert THUMB_W == 256
    assert StochasticEmitter(48000, 200)._stride_for_width(THUMB_W) == 2


def test_baker_streams_large_thumbnail_arrays_to_an_atomic_memmap():
    import utilities.convert_to_xy as baker

    original = baker.vectorize
    try:
        baker.vectorize = lambda *_args, **_kwargs: (
            [], [], np.full((4, 3, 3), 137, dtype=np.uint8))
        with tempfile.TemporaryDirectory() as root:
            src = Path(root) / "0_source"
            dst = Path(root) / "baked"
            src.mkdir()
            (src / "frame1.png").write_bytes(b"fixture")
            (src / "frame2.png").write_bytes(b"fixture")
            profile = dict(baker.PROFILES["tiny"])
            profile["thumb_width"] = THUMB_W
            result = baker.process_folder((src, dst, profile))
            assert result is None
            thumbs = np.load(dst / "thumbs.npy", mmap_mode="r")
            assert thumbs.shape == (2, 4, 3, 3)
            assert np.all(thumbs == 137)
            assert not list(dst.glob(".thumbs-*.npy"))
    finally:
        baker.vectorize = original


def test_triangle_mix_default_sequence_and_duty():
    scheduler = TriangleMixScheduler(0.5)
    assert [scheduler.next_mode() for _ in range(8)] == [
        "vector", "raster", "stochastic", "raster",
        "vector", "raster", "stochastic", "raster",
    ]


def test_triangle_mix_preserves_raster_duty_and_splits_remainder():
    for duty in (0.0, 0.2, 0.5, 0.75, 1.0):
        scheduler = TriangleMixScheduler(duty)
        modes = [scheduler.next_mode() for _ in range(10000)]
        counts = {mode: modes.count(mode)
                  for mode in ("vector", "raster", "stochastic")}
        assert abs(counts["raster"] / len(modes) - duty) <= 1 / len(modes)
        assert abs(counts["vector"] - counts["stochastic"]) <= 1


def test_triangle_mix_preview_window_contains_every_active_component():
    for duty in (0.0, 0.2, 0.5, 0.6, 0.75, 0.9, 1.0):
        for fields in (1, 2, 4):
            span = TriangleMixScheduler.coverage_traces(duty, fields)
            scheduler = TriangleMixScheduler(duty)
            modes = [scheduler.next_mode() for _ in range(500)]
            for start in range(len(modes) - span + 1):
                window = modes[start:start + span]
                if duty > 0:
                    assert window.count("raster") >= fields
                if duty < 1:
                    assert "vector" in window
                    assert "stochastic" in window


def test_image_update_does_not_reset_continuous_walk():
    emitter = StochasticEmitter(48000, 800, seed=5)
    emitter.emit(_portrait())
    count = emitter._count
    changed = _portrait()
    changed[:, :20] = 0.8
    emitter.emit(changed)
    assert count == 800
    assert emitter._count == 1600


def test_live_mode_cycles_all_four_renderers_and_fusion_sets():
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
    assert state["mode"] == "fusion"
    state["fusion_components"] = "vrs"
    assert "--scope-mode fusion" in as_flags(state)
    assert "--scope-fusion vrs" in as_flags(state)
    for expected in ("vr", "sv", "sr", "vrs"):
        keys.feed("f")
        assert state["fusion_components"] == expected
    keys.feed("v")
    assert state["mode"] == "vector"
    assert state["gamma"] == 2.2


def test_fusion_density_supports_every_requested_component_set():
    shape = (24, 30)
    vector = np.zeros(shape)
    vector[:, 14:17] = 1.0
    raster = np.zeros(shape)
    raster[4:20, 2:8] = 1.0
    stochastic = np.zeros(shape)
    stochastic[4:20, 22:28] = 1.0
    regions = {
        "v": np.s_[:, 14:17],
        "r": np.s_[4:20, 2:8],
        "s": np.s_[4:20, 22:28],
    }
    for combo in ("vrs", "vr", "sv", "sr"):
        fused = fuse_density(
            vector, raster, stochastic, components=combo,
            raster_gamma=1.0, stochastic_gamma=1.0)
        assert fused.shape == shape
        assert np.isclose(fused.max(), 1.0)
        masses = {name: float(fused[region].sum())
                  for name, region in regions.items()}
        for name in "vrs":
            assert (masses[name] > 0.0) == (name in combo)
        active = [masses[name] for name in "vrs" if name in combo]
        assert max(active) - min(active) < 1e-9


def test_vector_density_and_fusion_runtime_emit_all_combinations():
    import scope_display

    class Lib:
        thumbs = True

        def __len__(self):
            return 1

        def thumb(self, _index):
            thumb = np.zeros((64, 48, 3), dtype=np.uint8)
            thumb[10:54, 8:40, 0] = 170
            thumb[10:54, 8:40, 1] = 255
            thumb[18:46, 14:34, 2] = 220
            return thumb

        def frame(self, _index):
            square = np.array([
                [-0.55, -0.65], [0.55, -0.65], [0.55, 0.65],
                [-0.55, 0.65], [-0.55, -0.65]], dtype=np.float32)
            return [square], [1]

    class FakeScope:
        samplerate = 48000
        samples_per_frame = 800

        def __init__(self):
            self.frames = []

        def show_frame(self, frame):
            self.frames.append(frame)

    lib = Lib()
    vd = vector_density(lib.frame(0)[0], (64, 48))
    assert vd.shape == (64, 48) and vd.sum() > 0
    assert normalize_fusion_components("s+v") == "sv"

    scope_display.Scope._tap_until = 0
    scope = FakeScope()
    stochastic = StochasticEmitter(48000, 800, seed=22)
    beam = None
    for combo in ("vrs", "vr", "sv", "sr"):
        probability = fusion_probability(lib, 0, None, 0, components=combo)
        assert probability.shape == (64, 48)
        assert probability.max() == 1.0
        beam = scope_display._emit(
            scope, lib, None, 0, "fusion", {}, "alternate",
            2.2, 0.02, 1.0, None, 0.02,
            stochastic_emitter=stochastic, beam_start=beam,
            mode_handoff=True, fusion_components=combo,
            stochastic_gamma=2.0)
        frame = scope.frames[-1]
        assert frame.shape == (800, 2)
        assert np.isfinite(frame).all()


def test_live_mode_cycle_respects_fixed_scheduler():
    state = {"mode": "raster", "raster": True, "mode_locked": True}
    keys = KeyMap(state)
    keys.feed("v")
    assert state["mode"] == "raster"
    assert "unavailable" in keys.message


def test_mix_printed_flags_preserve_scheduler_and_both_gammas():
    state = {
        "mode": "vector", "mix_hz": 120.0, "mix_duty": 0.5,
        "trim": 0.02, "density": 1.0, "raster_gamma": 2.2,
        "stochastic_gamma": 2.0, "sweep": "alternate", "autofit": True,
    }
    flags = as_flags(state)
    assert "--scope-mix 120" in flags
    assert "--scope-mix-duty 0.5" in flags
    assert "--scope-gamma 2.2" in flags
    assert "--scope-stochastic-gamma 2" in flags


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
    raster_end = scope_display._emit(
        *common, "raster", {}, "alternate", 2.2, 0.02, 1.0, None, 0.02,
        lowpass=3000, emitter=raster, stochastic_emitter=stochastic,
        beam_start=continuous_end)
    assert np.array_equal(scope.frames[-1][0], continuous_end)
    scope_display._emit(
        *common, "stochastic", {}, "alternate", 2.2, 0.02, 1.0, None, 0.02,
        lowpass=3000, emitter=raster, stochastic_emitter=stochastic,
        beam_start=raster_end, mode_handoff=True)
    assert np.array_equal(scope.frames[-1][0], raster_end)
    # Two 800-sample stochastic passes reached count 1600. Re-entering after
    # raster spends one sample on the exact handoff, then advances 799 more;
    # it must not reset the persistent walk clock to one.
    assert stochastic._count == 2399


def test_stochastic_handoff_preserves_fractional_walk_clock():
    emitter = StochasticEmitter(96000, 7, walk_hz=44100, seed=3)
    emitter.emit(_portrait())
    phase = emitter._phase
    count = emitter._count
    emitter.start_at(np.array([0.25, -0.4], dtype=np.float32))
    assert emitter._phase == phase
    assert emitter._count == count


def test_scope_tap_publishes_one_complete_mixed_exposure():
    import time
    from scope_out import Scope

    original_tap = Scope._tap
    original_fields = Scope._tap_fields
    original_accum = Scope._tap_accum
    original_until = Scope._tap_until
    try:
        scope = object.__new__(Scope)
        Scope._tap = {"seq": 0, "data": None}
        Scope.set_tap_fields(4)
        Scope._tap_until = time.monotonic() + 1.0
        for i in range(3):
            scope._capture(np.full((5, 2), i, dtype=np.float32))
        assert Scope.read_tap() == (0, None)
        scope._capture(np.full((5, 2), 3, dtype=np.float32))
        seq, data = Scope.read_tap()
        assert seq == 1
        assert data.shape == (20, 2)
        assert np.allclose(data[::5, 0], np.arange(4) / 0.9)
    finally:
        Scope._tap = original_tap
        Scope._tap_fields = original_fields
        Scope._tap_accum = original_accum
        Scope._tap_until = original_until


def test_combined_256_bake_triangle_mix_emits_all_three_paths():
    import scope_display

    class Lib:
        thumbs = True

        def __len__(self):
            return 1

        def thumb(self, _index):
            thumb = np.zeros((341, 256, 3), dtype=np.uint8)
            thumb[70:290, 55:205, 0] = 150
            thumb[70:290, 55:205, 1] = 255
            thumb[120:230, 85:175, 2] = 230
            return thumb

        def frame(self, _index):
            square = np.array([
                [-0.55, -0.65], [0.55, -0.65], [0.55, 0.65],
                [-0.55, 0.65], [-0.55, -0.65]], dtype=np.float32)
            return [square], [1]

    class FakeScope:
        samplerate = 48000
        samples_per_frame = 400

        def __init__(self):
            self.frames = []

        def show_frame(self, frame):
            self.frames.append(frame)

    scope_display.Scope._tap_until = 0
    scope = FakeScope()
    lib = Lib()
    raster = TraceEmitter(48000, 400, fields=2, sweep="alternate")
    stochastic = StochasticEmitter(48000, 400, seed=12)
    scheduler = TriangleMixScheduler(0.5)
    sweep = {}
    beam = None
    last_mode = None
    raster_field = 0
    modes = []

    for _ in range(8):
        mode = scheduler.next_mode()
        modes.append(mode)
        end = scope_display._emit(
            scope, lib, None, 0, mode, sweep, "alternate",
            2.2, 0.02, 1.0, None, 0.02,
            emitter=raster, stochastic_emitter=stochastic,
            beam_start=beam,
            mode_handoff=(last_mode is not None and mode != last_mode),
            field=raster_field % 2, fields=2)
        frame = scope.frames[-1]
        assert frame.shape == (400, 2)
        assert np.isfinite(frame).all()
        if beam is not None and mode in ("raster", "stochastic"):
            assert np.array_equal(frame[0], beam)
        beam = end
        if mode == "raster":
            raster_field += 1
        last_mode = mode

    assert modes == ["vector", "raster", "stochastic", "raster"] * 2
    assert raster_field == 4
    assert stochastic._count > 400  # its clock survived the intervening modes
