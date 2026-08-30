"""Regression checks for no-Z stochastic, stipple, and fusion renderers."""
import tempfile
from pathlib import Path

import numpy as np

from scope_bake import (StochasticEmitter, StippleEmitter, TraceEmitter,
                        TriangleMixScheduler,
                        PositionMultiplexer,
                        calibrate, composite_luma,
                        composite_stipple_candidates, raster_precondition_for,
                        _precondition_grid, merge, stochastic_luma,
                        fuse_density, fuse_positions,
                        normalize_fusion_components, vector_density,
                        trace_luminance_weights, raster_frame)
from scope_bake import retime_trace_by_weights
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


def test_compact_thumbnail_default_stays_inside_old_storage_range():
    assert THUMB_W == 128
    assert StochasticEmitter(48000, 200)._stride_for_width(THUMB_W) == 1


def test_baker_streams_large_thumbnail_arrays_to_an_atomic_memmap():
    import utilities.convert_to_xy as baker

    original = baker.vectorize
    try:
        def fake_vectorize(*_args, **kwargs):
            count = kwargs["stipple_candidates"]
            return ([], [], np.full((4, 3, 2), 137, dtype=np.uint8),
                    np.full((count, 2), 1234, dtype=np.uint16),
                    np.full((count, 3), 99, dtype=np.uint8), np.float32(0.4))

        baker.vectorize = fake_vectorize
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
            assert thumbs.shape == (2, 4, 3, 2)
            assert np.all(thumbs == 137)
            assert not list(dst.glob(".thumbs-*.npy"))
            assert np.load(dst / "stipple_xy.npy", mmap_mode="r").shape == (
                2, baker.STIPPLE_CANDIDATES, 2)
            assert np.load(dst / "stipple_lae.npy", mmap_mode="r").shape == (
                2, baker.STIPPLE_CANDIDATES, 3)
            meta = __import__("json").loads((dst / "format.json").read_text())
            assert meta["thumbnail_channels"] == ["raw_luminance", "alpha"]
            assert meta["thumb_width"] == 128
            from scope_bake import XYLibrary
            lib = XYLibrary(dst)
            assert lib.raw_thumbnail
            assert lib.raster_precondition == baker.PRECONDITION
            assert lib.stipple(1)[0].shape == (baker.STIPPLE_CANDIDATES, 2)
    finally:
        baker.vectorize = original


def test_triangle_mix_default_sequence_and_duty():
    scheduler = TriangleMixScheduler(0.5)
    assert [scheduler.next_mode() for _ in range(12)] == [
        "vector", "raster", "stochastic", "raster", "stipple", "raster",
        "vector", "raster", "stochastic", "raster", "stipple", "raster",
    ]


def test_triangle_mix_preserves_raster_duty_and_splits_remainder():
    for duty in (0.0, 0.2, 0.5, 0.75, 1.0):
        scheduler = TriangleMixScheduler(duty)
        modes = [scheduler.next_mode() for _ in range(10000)]
        counts = {mode: modes.count(mode)
                  for mode in ("vector", "raster", "stochastic", "stipple")}
        assert abs(counts["raster"] / len(modes) - duty) <= 1 / len(modes)
        assert abs(counts["vector"] - counts["stochastic"]) <= 1
        assert abs(counts["vector"] - counts["stipple"]) <= 1


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
                    assert "stipple" in window


def test_image_update_does_not_reset_continuous_walk():
    emitter = StochasticEmitter(48000, 800, seed=5)
    emitter.emit(_portrait())
    count = emitter._count
    changed = _portrait()
    changed[:, :20] = 0.8
    emitter.emit(changed)
    assert count == 800
    assert emitter._count == 1600


def _assert_full_border(frame, aspect, level=0.9):
    sx = level if aspect <= 1.0 else level / aspect
    sy = level * aspect if aspect <= 1.0 else level
    expected = ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy))
    for corner in expected:
        assert np.any(np.all(np.isclose(frame, corner, atol=1e-6), axis=1))


def test_stochastic_border_preserves_walk_clock_and_hidden_endpoint():
    lum = _portrait()
    plain = StochasticEmitter(48000, 800, seed=19)
    boxed = StochasticEmitter(48000, 800, seed=19, border=0.04)
    for _ in range(2):
        a = plain.emit(lum)
        b = boxed.emit(lum)
        assert a.shape == b.shape == (800, 2)
        assert np.array_equal(a[-1], b[-1])
        assert plain._count == boxed._count
        assert plain._pixel == boxed._pixel
        _assert_full_border(b, lum.shape[0] / lum.shape[1])


def test_sparse_stipple_border_preserves_route_endpoint_between_frames():
    cloud = {
        "xy": np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]]),
        "luminance": np.array([0.25, 0.6, 1.0]),
        "edge": np.zeros(3),
        "correction": np.ones(3),
        "aspect": 4 / 3,
    }
    plain = StippleEmitter(48000, 800, points=96)
    boxed = StippleEmitter(48000, 800, points=96, border=0.04)
    for _ in range(2):
        a = plain.emit_candidates(cloud)
        b = boxed.emit_candidates(cloud)
        assert np.array_equal(a[-1], b[-1])
        _assert_full_border(b, cloud["aspect"])


def test_stipple_is_stable_euclidean_and_rate_independent():
    lum = _portrait()
    a = StippleEmitter(48000, 800, points=192).emit(lum)
    b = StippleEmitter(48000, 800, points=192).emit(lum)
    hi = StippleEmitter(96000, 1600, points=192).emit(lum)
    assert np.array_equal(a, b)
    assert a.shape == (800, 2) and hi.shape == (1600, 2)
    assert np.isfinite(a).all() and np.abs(a).max() <= 0.900001
    # The same image route is merely sampled more densely at the larger output
    # array length; compare both at common normalized positions.
    common = np.linspace(0.0, 1.0, 200)
    ta = np.linspace(0.0, 1.0, len(a))
    th = np.linspace(0.0, 1.0, len(hi))
    for axis in range(2):
        assert np.allclose(np.interp(common, ta, a[:, axis]),
                           np.interp(common, th, hi[:, axis]), atol=1e-2)
    delta = np.diff(a, axis=0)
    assert np.any((np.abs(delta[:, 0]) > 1e-5)
                  & (np.abs(delta[:, 1]) > 1e-5))


def test_sparse_stipple_candidates_keep_source_coordinates_and_live_gamma():
    cloud = {
        "xy": np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]]),
        "luminance": np.array([0.25, 0.6, 1.0]),
        "edge": np.zeros(3),
        "correction": np.ones(3),
        "aspect": 4 / 3,
    }
    a = StippleEmitter(48000, 800, points=96, gamma=1.0).emit_candidates(cloud)
    b = StippleEmitter(48000, 800, points=96, gamma=1.0).emit_candidates(cloud)
    dark = StippleEmitter(48000, 800, points=96, gamma=4.0).emit_candidates(cloud)
    assert np.array_equal(a, b)
    assert a.shape == dark.shape == (800, 2)
    assert np.isfinite(a).all() and np.abs(a).max() <= 0.900001
    assert not np.array_equal(a, dark)


def test_grid_precondition_is_horizontal_clamped_and_face_safe():
    base = np.array([[0.1, 0.4, 0.2, 0.8, 0.3],
                     [0.2, 0.6, 0.3, 0.7, 0.2],
                     [0.9, 0.1, 0.8, 0.2, 0.9]], dtype=np.float64)
    changed_vertical_neighbours = base.copy()
    changed_vertical_neighbours[0] = 1.0 - base[0]
    changed_vertical_neighbours[2] = 1.0 - base[2]
    actual = _precondition_grid(base, 0.45)
    other = _precondition_grid(changed_vertical_neighbours, 0.45)
    assert np.allclose(actual[1], other[1])
    for row_in, row_out in zip(base, actual):
        padded = np.pad(row_in, 1, mode="edge")
        assert np.all(row_out >= np.minimum.reduce(
            [padded[:-2], padded[1:-1], padded[2:]]) - 1e-12)
        assert np.all(row_out <= np.maximum.reduce(
            [padded[:-2], padded[1:-1], padded[2:]]) + 1e-12)


def test_new_bake_metadata_enables_grid_precondition_only_once():
    class New:
        raw_thumbnail = True
        raster_precondition = 0.45

    class Legacy:
        raw_thumbnail = False
        raster_precondition = 0.0

    assert raster_precondition_for(New(), New()) == 0.45
    assert raster_precondition_for(New(), Legacy()) == 0.0


def test_raster_default_ignores_old_sharpening_metadata_without_losing_grid():
    class Lib:
        raw_thumbnail = True
        raster_precondition = 0.45

        def __len__(self):
            return 1

        def thumb(self, _index):
            thumb = np.zeros((48, 36, 2), dtype=np.uint8)
            thumb[5:43, 4:32, 0] = np.tile(
                np.array([40, 100, 180, 240], dtype=np.uint8), (38, 7))
            thumb[5:43, 4:32, 1] = 255
            return thumb

    natural = raster_frame(Lib(), 0, None, 0, 800)
    explicit_zero = raster_frame(Lib(), 0, None, 0, 800, precondition=0.0)
    sharpened = raster_frame(Lib(), 0, None, 0, 800, precondition=0.45)
    assert np.array_equal(natural, explicit_zero)
    assert natural.shape == sharpened.shape == (800, 2)
    assert not np.array_equal(natural, sharpened)


def test_sparse_stipple_composite_respects_float_over_main_alpha():
    class Lib:
        def __init__(self, lum, alpha):
            self._thumb = np.zeros((4, 4, 2), dtype=np.uint8)
            self._thumb[..., 0] = lum
            self._thumb[..., 1] = alpha
            self._xy = np.array([[32768, 32768]], dtype=np.uint16)
            self._lae = np.array([[lum, alpha, 0]], dtype=np.uint8)

        def __len__(self):
            return 1

        def thumb(self, _index):
            return self._thumb

        def stipple(self, _index):
            return self._xy, self._lae, 0.5

    main = Lib(200, 255)
    floating = Lib(100, 255)
    cloud = composite_stipple_candidates(main, 0, floating, 0)
    # First pool is main and is fully occluded; second is the float layer.
    assert cloud["luminance"][0] == 0.0
    assert np.isclose(cloud["luminance"][1], 100 / 255)
    inverse = composite_stipple_candidates(
        main, 0, floating, 0, invert=True)
    assert inverse["luminance"][0] == 0.0
    assert np.isclose(inverse["luminance"][1], 1.0 - 100 / 255)


def test_inverse_luminance_preserves_transparent_padding():
    class Lib:
        def __init__(self):
            self._thumb = np.zeros((2, 3, 2), dtype=np.uint8)
            self._thumb[..., 0] = np.array(
                [[0, 64, 255], [255, 128, 0]], dtype=np.uint8)
            self._thumb[..., 1] = np.array(
                [[0, 255, 255], [128, 128, 0]], dtype=np.uint8)

        def __len__(self):
            return 1

        def thumb(self, _index):
            return self._thumb

    normal = composite_luma(Lib(), 0, None, 0)
    inverse = composite_luma(Lib(), 0, None, 0, invert=True)
    alpha = Lib().thumb(0)[..., 1].astype(np.float64) / 255.0
    assert np.allclose(normal + inverse, alpha)
    assert inverse[0, 0] == 0.0       # transparent black does not turn white
    assert inverse[1, 2] == 0.0


def test_live_mode_cycles_all_five_renderers_and_fusion_sets():
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
    assert state["mode"] == "stipple"
    assert state["gamma"] == 2.0
    assert "--scope-mode stipple" in as_flags(state)
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


def test_live_inverse_toggle_and_printed_flag():
    state = {"mode": "raster", "raster": True, "invert": False}
    keys = KeyMap(state)
    assert keys.feed("i")
    assert state["invert"] is True
    assert keys.dirty
    assert "--scope-invert" in as_flags(state)
    keys.dirty = False
    keys.feed("i")
    assert state["invert"] is False
    assert "--scope-invert" not in as_flags(state)


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


def test_position_fusion_multiplexes_corresponding_array_entries():
    vector = np.column_stack([np.arange(8), np.full(8, 10)])
    raster = np.column_stack([np.arange(8), np.full(8, 20)])
    stochastic = np.column_stack([np.arange(8), np.full(8, 30)])
    sources = {"v": vector, "r": raster, "s": stochastic}
    for combo in ("vrs", "vr", "sv", "sr"):
        mux = PositionMultiplexer()
        actual = mux.emit(vector, raster, stochastic, components=combo)
        expected = np.empty_like(vector, dtype=np.float32)
        for i in range(len(expected)):
            expected[i] = sources[combo[i % len(combo)]][i]
        assert actual.dtype == np.float32
        assert np.array_equal(actual, expected)

    # Eight entries do not divide evenly across V/R/S. The next array resumes
    # at S instead of favoring V at the beginning of every call.
    mux = PositionMultiplexer()
    mux.emit(vector, raster, stochastic, components="vrs")
    continued = mux.emit(vector, raster, stochastic, components="vrs")
    assert np.array_equal(continued[0], stochastic[0])


def test_position_fusion_allocates_more_samples_to_brighter_candidates():
    n = 1000
    vector = np.column_stack([np.arange(n), np.full(n, 10)])
    raster = np.column_stack([np.arange(n), np.full(n, 20)])
    weights = {
        "v": np.full(n, 0.9),
        "r": np.full(n, 0.1),
    }
    actual = PositionMultiplexer().emit(
        vector=vector, raster=raster, components="vr", weights=weights)
    vector_count = int(np.count_nonzero(actual[:, 1] == 10))
    raster_count = int(np.count_nonzero(actual[:, 1] == 20))
    assert vector_count == 900
    assert raster_count == 100

    # With no light at either candidate, selection remains balanced rather
    # than parking on whichever component happens to be first.
    dark = {"v": np.zeros(n), "r": np.zeros(n)}
    fallback = PositionMultiplexer().emit(
        vector=vector, raster=raster, components="vr", weights=dark)
    assert np.count_nonzero(fallback[:, 1] == 10) == n // 2


def test_trace_luminance_weights_distinguish_light_and_dark_positions():
    lum = np.zeros((32, 32), dtype=np.float64)
    lum[:, 16:] = 1.0
    trace = np.array([[-0.6, 0.0], [0.6, 0.0]], dtype=np.float32)
    weights = trace_luminance_weights(lum, trace, gamma=2.0, trim=0.02)
    assert weights[0] == 0.0
    assert weights[1] > 0.99


def test_inverse_vector_retiming_preserves_geometry_and_favors_dark_source():
    x = np.linspace(-0.9, 0.9, 401, dtype=np.float32)
    trace = np.column_stack([x, np.zeros_like(x)])
    # This is already inverse luminance: the source's dark left side is now
    # high weight and should receive most of the vector's beam time.
    weights = np.where(x < 0.0, 1.0, 0.05).astype(np.float32)
    inverse = retime_trace_by_weights(trace, weights)
    assert inverse.shape == trace.shape
    assert np.array_equal(inverse[0], trace[0])
    assert np.array_equal(inverse[-1], trace[-1])
    assert np.count_nonzero(inverse[:, 0] < 0.0) > 300
    assert np.all(np.diff(inverse[:, 0]) >= -1e-6)


def test_runtime_vector_mode_applies_inverse_dwell_retiming():
    import scope_display

    class Lib:
        def __len__(self):
            return 1

        def thumb(self, _index):
            thumb = np.zeros((32, 64, 2), dtype=np.uint8)
            thumb[..., 0] = np.linspace(0, 255, 64, dtype=np.uint8)
            thumb[..., 1] = 255
            return thumb

        def frame(self, _index):
            line = np.array([
                [-0.8, -0.2], [0.8, -0.2], [0.8, 0.2], [-0.8, 0.2],
            ], dtype=np.float32)
            return [line], [0]

    class Scope:
        samples_per_frame = 800
        samplerate = 48000

        def __init__(self):
            self.frames = []

        def show_frame(self, frame):
            self.frames.append(frame.copy())

    scope_display.Scope._tap_until = 0
    scope = Scope()
    args = (scope, Lib(), None, 0, "vector", {}, "alternate",
            2.2, 0.02, 1.0, None, 0.02)
    scope_display._emit(*args, invert=False)
    normal = scope.frames[-1]
    scope_display._emit(*args, invert=True)
    inverse = scope.frames[-1]
    assert normal.shape == inverse.shape == (800, 2)
    assert np.array_equal(normal[0], inverse[0])
    assert np.array_equal(normal[-1], inverse[-1])
    assert not np.array_equal(normal, inverse)


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
    raster = TraceEmitter(48000, 800)
    from scope_out import rasterize
    expected_vector = rasterize(merge(lib, 0, None, 0), 800)
    expected_raster = TraceEmitter(48000, 800).emit(
        composite_luma(lib, 0, None, 0))
    expected_stochastic = StochasticEmitter(48000, 800, seed=22).emit(
        composite_luma(lib, 0, None, 0, raw=True))
    fusion_luma = composite_luma(lib, 0, None, 0, raw=True)
    expected_weights = {
        "v": trace_luminance_weights(
            fusion_luma, expected_vector, gamma=2.2, trim=0.02),
        "r": trace_luminance_weights(
            fusion_luma, expected_raster, gamma=2.2, trim=0.02),
        "s": trace_luminance_weights(
            fusion_luma, expected_stochastic, gamma=2.0, trim=0.02),
    }
    expected_vrs = fuse_positions(
        expected_vector, expected_raster, expected_stochastic, "vrs",
        weights=expected_weights)
    beam = None
    for combo in ("vrs", "vr", "sv", "sr"):
        beam = scope_display._emit(
            scope, lib, None, 0, "fusion", {}, "alternate",
            2.2, 0.02, 1.0, None, 0.02,
            emitter=raster, stochastic_emitter=stochastic, beam_start=beam,
            mode_handoff=True, fusion_components=combo,
            stochastic_gamma=2.0)
        frame = scope.frames[-1]
        assert frame.shape == (800, 2)
        assert np.isfinite(frame).all()
        if combo == "vrs":
            assert np.allclose(frame, expected_vrs)

    scope_display._emit(
        scope, lib, None, 0, "fusion", {}, "alternate",
        2.2, 0.02, 1.0, None, 0.02,
        border=0.04, emitter=raster, stochastic_emitter=stochastic,
        beam_start=beam, mode_handoff=True, fusion_components="vr",
        stochastic_gamma=2.0)
    _assert_full_border(scope.frames[-1], 64 / 48)


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


def test_baked_manifest_matches_normal_folder_count_grouping_and_order():
    import scope_display

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def bake(rel, frames):
            dest = root / rel
            dest.mkdir(parents=True)
            np.save(dest / "frame_starts.npy",
                    np.arange(frames + 1, dtype=np.int32))
            return dest

        m0 = bake("face/0_base", 4)
        m2 = bake("face/2_second", 4)
        bake("face/1_wrong_length", 7)
        m10 = bake("face/10_tenth", 4)
        f2 = bake("float/255_2_float", 4)
        f10 = bake("float/255_10_float", 4)
        bake("float/255_wrong_length", 6)

        frames, mains, floats = scope_display._manifest_from_xy(root)
        assert frames == 4
        assert mains == [m0, m2, m10]
        assert floats == [f2, f10]

        # The interactive pair tester must offer exactly the libraries the
        # live runtime can select, rather than every stale bake on disk.
        import test_scope_pair
        pair_mains, pair_floats, pair_root = test_scope_pair.discover(root)
        assert pair_root == root
        assert pair_mains == mains
        assert pair_floats == floats


def test_pair_mixer_playback_is_not_gated_on_gui():
    """Headless/audio-only playback must advance beyond its first V trace."""
    import inspect
    import test_scope_pair

    source = inspect.getsource(test_scope_pair.main)
    assert "if playing and gui:" not in source
    assert "scope is None or scope.ready()" in source


def test_legacy_three_channel_bake_mix_emits_all_four_paths():
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
    stipple = StippleEmitter(48000, 400, points=96)
    scheduler = TriangleMixScheduler(0.5)
    sweep = {}
    beam = None
    last_mode = None
    raster_field = 0
    modes = []

    for _ in range(12):
        mode = scheduler.next_mode()
        modes.append(mode)
        end = scope_display._emit(
            scope, lib, None, 0, mode, sweep, "alternate",
            2.2, 0.02, 1.0, None, 0.02,
            emitter=raster, stochastic_emitter=stochastic,
            stipple_emitter=stipple,
            beam_start=beam,
            mode_handoff=(last_mode is not None and mode != last_mode),
            field=raster_field % 2, fields=2)
        frame = scope.frames[-1]
        assert frame.shape == (400, 2)
        assert np.isfinite(frame).all()
        if beam is not None and mode in ("raster", "stochastic", "stipple"):
            assert np.array_equal(frame[0], beam)
        beam = end
        if mode == "raster":
            raster_field += 1
        last_mode = mode

    assert modes == [
        "vector", "raster", "stochastic", "raster", "stipple", "raster",
    ] * 2
    assert raster_field == 6
    assert stochastic._count > 400  # its clock survived the intervening modes
