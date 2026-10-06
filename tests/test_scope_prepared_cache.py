"""Prepared-image reuse must preserve pixels and accepted trajectory state."""
import numpy as np

from scope_bake import (StippleEmitter, SweepSource, TraceEmitter,
                        _stochastic_probability, composite_luma,
                        composite_stipple_candidates, prepare_render_grid)
from scope_prepared_cache import PreparedImageCache


class Library:
    def __init__(self, versioned=True):
        if versioned:
            self.prepared_source_version = object()
        self.images = np.zeros((2, 16, 20, 3), dtype=np.uint8)
        self.images[:, :, :, 1] = 255
        self.images[0, 2:12, 3:17, 0] = 190
        self.images[1, 4:14, 1:12, 0] = 240
        self.images[:, :, :, 2] = 75
        self.calls = 0

    def __len__(self):
        return len(self.images)

    def thumb(self, index):
        self.calls += 1
        return self.images[index]


class CandidateLibrary(Library):
    def __init__(self, versioned=True):
        super().__init__(versioned=versioned)
        self.candidate_reads = 0
        self.candidate_xy = np.array(
            [[10000, 10000], [22000, 13000], [42000, 32000],
             [53000, 54000]], dtype=np.uint16)
        self.candidate_lae = np.array(
            [[220, 255, 80], [170, 255, 40], [245, 255, 90],
             [130, 255, 20]], dtype=np.uint8)

    def stipple(self, _index):
        self.candidate_reads += 1
        return self.candidate_xy, self.candidate_lae, 0.8


class GeometryLibrary(Library):
    def __init__(self):
        super().__init__()
        self.geometry_reads = 0

    def frame(self, index):
        self.geometry_reads += 1
        offset = 0.1 * int(index)
        line = np.array([[-0.7 + offset, -0.4], [0.2, 0.6],
                         [0.7 - offset, -0.3]], dtype=np.float32)
        return [line], [0]


def test_keys_cover_sources_versions_indices_crop_raw_and_inversion():
    main, floating = Library(), Library()
    floating.images[:, :, :, 1] = 128
    cache = PreparedImageCache()
    for index, raw, invert, bbox, rotation in (
            (0, False, False, None, 0), (1, False, False, None, 0),
            (0, True, False, None, 0), (0, False, True, None, 0),
            (0, False, False, (0.1, 0.2, 0.8, 0.9), 0),
            (0, False, False, None, 90)):
        expected = composite_luma(main, index, floating, 0,
                                  raw=raw, invert=invert, bbox=bbox)
        if rotation:
            expected = np.ascontiguousarray(np.rot90(expected, rotation // 90))
        image = cache.luma(main, index, floating, 0,
                           raw=raw, invert=invert, bbox=bbox,
                           rotation=rotation)
        np.testing.assert_array_equal(image, expected)
        calls = main.calls + floating.calls
        assert cache.luma(main, index, floating, 0,
                          raw=raw, invert=invert, bbox=bbox,
                          rotation=rotation) is image
        assert main.calls + floating.calls == calls
        assert not image.flags.writeable
    old = cache.luma(main, 0, floating, 0)
    main.images[0, :, :, 0] = 30
    main.prepared_source_version = object()
    refreshed = cache.luma(main, 0, floating, 0)
    assert not np.array_equal(old, refreshed)
    other = Library()
    cache.luma(other, 0, floating, 0)
    assert cache.snapshot()["misses"] == 8


def test_stochastic_probability_cache_is_setting_and_source_versioned():
    main = Library()
    cache = PreparedImageCache()
    lum = cache.luma(main, 0, None, 0, raw=True)
    key = cache.source_key(main, 0, None, 0, raw=True)
    expected = _stochastic_probability(lum, 1.7, 0.08, 0.4)
    probability = cache.stochastic_probability(
        lum, key, gamma=1.7, trim=0.08, edge_gain=0.4)
    np.testing.assert_array_equal(probability, expected)
    assert cache.stochastic_probability(
        lum, key, gamma=1.7, trim=0.08, edge_gain=0.4) is probability
    assert not probability.flags.writeable
    tuned = cache.stochastic_probability(
        lum, key, gamma=2.1, trim=0.08, edge_gain=0.4)
    assert tuned is not probability
    np.testing.assert_array_equal(
        tuned, _stochastic_probability(lum, 2.1, 0.08, 0.4))

    live = Library(versioned=False)
    live_key = cache.source_key(live, 0, None, 0, raw=True)
    before = cache.stochastic_probability(
        live.thumb(0)[..., 0] / 255.0, live_key,
        gamma=2.0, trim=0.02, edge_gain=0.0)
    changed = live.images[0].copy()
    changed[..., 0] = 255
    after = cache.stochastic_probability(
        changed[..., 0] / 255.0, live_key,
        gamma=2.0, trim=0.02, edge_gain=0.0)
    assert not np.array_equal(before, after)
    assert cache.snapshot()["bypasses"] == 2


def test_stochastic_sparse_fallback_cdf_matches_uncached_random_walk():
    from scope_bake import StochasticEmitter

    class TrackingEmitter(StochasticEmitter):
        def __init__(self, *args, **kwargs):
            self.fallbacks = 0
            super().__init__(*args, **kwargs)

        def _find_white(self, probability, cdf=None):
            self.fallbacks += 1
            return super()._find_white(probability, cdf=cdf)

    probability = np.zeros((32, 48), dtype=np.float64)
    probability[17, 29] = 0.8
    source = Library()
    key = PreparedImageCache.source_key(source, 0, None, 0, raw=True)
    cache = PreparedImageCache()
    reference = TrackingEmitter(
        48000, 512, seed=17, radius=1, stride=1, reseed_ms=0.0)
    cached = TrackingEmitter(
        48000, 512, seed=17, radius=1, stride=1, reseed_ms=0.0)
    expected = reference.emit_probability(probability)

    cdf_provider = lambda: cache.stochastic_cdf(
        probability, key, gamma=2.0, trim=0.02, edge_gain=0.0)
    actual = cached.emit_probability(probability, cdf=cdf_provider)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(cached._end, reference._end)
    assert cached.fallbacks > 0 and reference.fallbacks == cached.fallbacks
    assert cache.snapshot()["entries"] == 1


def test_stipple_candidate_and_sample_cache_leave_endpoint_tour_mutable():
    main = CandidateLibrary()
    cache = PreparedImageCache(max_bytes=200_000)
    cloud = cache.stipple_candidates(main, 0, None, 0, rotation=90)
    reference_cloud = composite_stipple_candidates(main, 0, None, 0)
    # Rotation is image-space geometry and is included in candidate provenance.
    from scope_display import _rotate_stipple_cloud
    expected_cloud = _rotate_stipple_cloud(reference_cloud, 90)
    for name in ("xy", "luminance", "edge", "correction"):
        np.testing.assert_array_equal(cloud[name], expected_cloud[name])
    assert cloud["aspect"] == expected_cloud["aspect"]
    reads = main.candidate_reads
    assert cache.stipple_candidates(main, 0, None, 0, rotation=90) is cloud
    assert main.candidate_reads == reads

    source_key = cache.source_key(
        main, 0, None, 0, raw=True, rotation=90)
    samples = cache.stipple_candidate_samples(
        cloud, source_key, points=64, gamma=1.3, trim=0.01, edge_gain=0.2)
    direct = StippleEmitter(48000, 512, points=64, gamma=1.3,
                            trim=0.01, edge_gain=0.2)
    cached = StippleEmitter(48000, 512, points=64, gamma=1.3,
                            trim=0.01, edge_gain=0.2)
    direct.start_at((-0.5, 0.4))
    cached.start_at((-0.5, 0.4))
    cached_frame = cached.emit_candidates(
        cloud, prepared_samples=samples, tour_cache=cache,
        source_key=source_key)
    np.testing.assert_array_equal(cached_frame, direct.emit_candidates(cloud))
    same_start = StippleEmitter(48000, 512, points=64, gamma=1.3,
                                trim=0.01, edge_gain=0.2)
    same_start.start_at((-0.5, 0.4))
    np.testing.assert_array_equal(
        same_start.emit_candidates(
            cloud, prepared_samples=samples, tour_cache=cache,
            source_key=source_key),
        cached_frame)
    # Reusing sampled pixels must not freeze the endpoint-dependent greedy tour.
    first = StippleEmitter(48000, 512, points=64, gamma=1.3,
                           trim=0.01, edge_gain=0.2)
    second = StippleEmitter(48000, 512, points=64, gamma=1.3,
                            trim=0.01, edge_gain=0.2)
    first.start_at((-0.8, -0.7))
    second.start_at((0.8, 0.7))
    a = first.emit_candidates(
        cloud, prepared_samples=samples, tour_cache=cache,
        source_key=source_key)
    b = second.emit_candidates(
        cloud, prepared_samples=samples, tour_cache=cache,
        source_key=source_key)
    assert not np.array_equal(a, b)
    assert cache.stipple_candidate_samples(
        cloud, source_key, points=64, gamma=1.3,
        trim=0.01, edge_gain=0.2) is samples
    tone_changed = cache.stipple_candidate_samples(
        cloud, source_key, points=64, gamma=1.8,
        trim=0.01, edge_gain=0.2)
    assert tone_changed is not samples
    assert cache.snapshot()["hits"] >= 2
    assert cache.snapshot()["by_kind"]["stipple-tour"]["admissions"] == 1


def test_vector_geometry_cache_is_versioned_and_sample_budgeted():
    from scope_bake import merge
    from scope_out import rasterize, rotate_frame

    lib = GeometryLibrary()
    cache = PreparedImageCache()
    expected = rotate_frame(rasterize(merge(lib, 0, None, 0), 512), 270)
    frame = cache.vector_frame(
        lib, 0, None, 0, samples=512, min_feature=0.01, rotation=270)
    np.testing.assert_array_equal(frame, expected)
    assert frame.flags.writeable
    assert cache.snapshot()["entries"] == 0
    first_reads = lib.geometry_reads
    cached_frame = cache.vector_frame(
        lib, 0, None, 0, samples=512, min_feature=0.01,
        rotation=270)
    np.testing.assert_array_equal(cached_frame, frame)
    assert cached_frame is not frame
    assert not cached_frame.flags.writeable
    assert lib.geometry_reads == first_reads + 1
    reads = lib.geometry_reads
    assert cache.vector_frame(
        lib, 0, None, 0, samples=512, min_feature=0.01,
        rotation=270) is cached_frame
    assert lib.geometry_reads == reads
    assert cache.snapshot()["by_kind"]["vector-frame"]["admissions"] == 1

    changed = cache.vector_frame(
        lib, 1, None, 0, samples=512, min_feature=0.01, rotation=270)
    assert not np.array_equal(changed, frame)
    lib.prepared_source_version = object()
    refreshed = cache.vector_frame(
        lib, 0, None, 0, samples=512, min_feature=0.01, rotation=270)
    assert refreshed is not frame
    lower_rate = cache.vector_frame(
        lib, 0, None, 0, samples=256, min_feature=0.01, rotation=270)
    assert lower_rate.shape == (256, 2)
    assert lower_rate is not refreshed
    assert cache.snapshot()["bytes"] <= cache.snapshot()["max_bytes"]


def test_probation_is_bounded_and_admits_only_repeated_keys():
    cache = PreparedImageCache(max_bytes=1024, max_entries=3)
    for index in range(20):
        cache._prepared("vector-frame", index, (),
                        lambda: np.array([index], dtype=np.float32),
                        admit_after=2)
    assert len(cache._entries) == 0
    assert len(cache._probation) == 3
    assert cache.snapshot()["by_kind"]["vector-frame"]["probation_evictions"] == 17

    retained = cache._prepared(
        "vector-frame", 19, (), lambda: np.array([19], dtype=np.float32),
        admit_after=2)
    assert retained[0] == 19
    assert len(cache._entries) == 1
    assert cache.snapshot()["by_kind"]["vector-frame"]["admissions"] == 1


def test_unversioned_live_source_is_recomputed_and_memory_is_bounded():
    live = Library(versioned=False)
    cache = PreparedImageCache(max_bytes=2560, max_entries=1)
    before = cache.luma(live, 0, None, 0)
    live.images[0, :, :, 0] = 255
    after = cache.luma(live, 0, None, 0)
    assert not np.array_equal(before, after)
    assert cache.snapshot()["entries"] == 0
    assert cache.snapshot()["bypasses"] == 2
    baked = Library()
    cache.luma(baked, 0, None, 0)
    cache.luma(baked, 1, None, 0)
    stats = cache.snapshot()
    assert stats["bytes"] <= stats["max_bytes"]
    assert stats["entries"] == 1 and stats["evictions"] == 1
    assert stats["by_kind"]["luma"]["evictions"] == 1
    tiny = PreparedImageCache(max_bytes=1)
    tiny.luma(baked, 0, None, 0)
    assert tiny.snapshot()["bytes"] == 0


def test_unhashable_version_contract_bypasses_instead_of_raising():
    lib = Library()
    lib.prepared_source_version = ["mutable-version"]
    cache = PreparedImageCache()
    image = cache.luma(lib, 0, None, 0)
    assert image is not None
    assert cache.snapshot()["bypasses"] == 1
    assert PreparedImageCache.versioned_array_key(["mutable-version"]) is None


def test_cache_snapshot_reports_per_preparation_kind():
    lib = Library()
    cache = PreparedImageCache()
    lum = cache.luma(lib, 0, None, 0)
    cache.luma(lib, 0, None, 0)
    key = cache.source_key(lib, 0, None, 0)
    cache.stochastic_probability(
        lum, key, gamma=2.0, trim=0.02, edge_gain=0.0)
    live = Library(versioned=False)
    cache.luma(live, 0, None, 0)

    by_kind = cache.snapshot()["by_kind"]
    assert {key: by_kind["luma"][key] for key in (
        "hits", "misses", "bypasses", "evictions")} == {
            "hits": 1, "misses": 1, "bypasses": 1, "evictions": 0}
    assert {key: by_kind["stochastic-probability"][key] for key in (
        "hits", "misses", "bypasses", "evictions")} == {
            "hits": 0, "misses": 1, "bypasses": 0, "evictions": 0}


def test_cached_and_uncached_rendering_match_fields_endpoints_and_samples():
    lib = Library()
    cache = PreparedImageCache()
    reference = TraceEmitter(48000, 640, fields=2, sweep="alternate")
    cached = TraceEmitter(48000, 640, fields=2, sweep="alternate")
    for index in (0, 0, 1, 1, 0, 0):
        expected = reference.emit(composite_luma(lib, index, None, 0))
        actual = cached.emit(cache.luma(lib, index, None, 0))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(cached._end, reference._end)
        assert cached._field == reference._field
        assert cached._rev == reference._rev
    assert cache.snapshot()["hits"] == 4


def test_raster_grid_cache_tracks_render_settings_and_preserves_field_output():
    lib = Library()
    cache = PreparedImageCache(max_bytes=100_000)
    reference = TraceEmitter(48000, 640, fields=2, sweep="alternate")
    cached = TraceEmitter(48000, 640, fields=2, sweep="alternate")
    for index in (0, 0, 1, 1, 0, 0):
        lum = cache.luma(lib, index, None, 0)
        key = cache.source_key(lib, index, None, 0)
        grid = cache.raster_grid(
            lum, key, n=640, density=1.0, trim=0.02, rows=None,
            cols=None, autofit=True, grid_rows=None, grid_cols=None,
            row_bias=1.0, levels=None, stretch=True, fields=2,
            precondition=0.0, yt_fixed=False)
        expected_grid = prepare_render_grid(
            lum, 640, density=1.0, trim=0.02, rows=None, cols=None,
            autofit=True, grid_rows=None, grid_cols=None, row_bias=1.0,
            levels=None, stretch=True, fields=2, precondition=0.0,
            yt_fixed=False)
        for got, expected in zip(grid, expected_grid):
            np.testing.assert_array_equal(got, expected)
        actual = cached.emit(lum, prepared_grid=grid)
        expected = reference.emit(lum)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(cached._end, reference._end)
        assert cached._field == reference._field
        assert cached._rev == reference._rev

    assert cache.snapshot()["hits"] == 8  # 4 luma hits + 4 grid hits
    key = cache.source_key(lib, 0, None, 0)
    lum = composite_luma(lib, 0, None, 0)
    changed_rate = cache.raster_grid(
        lum, key, n=641, density=1.0, trim=0.02, rows=None, cols=None,
        autofit=True, grid_rows=None, grid_cols=None, row_bias=1.0,
        levels=None, stretch=True, fields=2, precondition=0.0,
        yt_fixed=False)
    grid_640 = cache.raster_grid(
        lum, key, n=640, density=1.0, trim=0.02, rows=None, cols=None,
        autofit=True, grid_rows=None, grid_cols=None, row_bias=1.0,
        levels=None, stretch=True, fields=2, precondition=0.0,
        yt_fixed=False)
    assert changed_rate is not grid_640
    sharpened = cache.raster_grid(
        lum, key, n=640, density=1.0, trim=0.02, rows=None, cols=None,
        autofit=True, grid_rows=None, grid_cols=None, row_bias=1.0,
        levels=None, stretch=True, fields=2, precondition=0.3,
        yt_fixed=False)
    assert sharpened is not grid_640
    stats = cache.snapshot()
    assert stats["bytes"] <= stats["max_bytes"]


def test_sweep_source_reuses_only_explicitly_versioned_unchanged_live_grids():
    class LiveSource:
        def __init__(self):
            self.image = np.zeros((24, 32), dtype=np.float32)
            self.image[4:20, 7:25] = 0.8
            self.version = object()

        def prepared_snapshot(self):
            return self.image, self.version

    source = LiveSource()
    generator = SweepSource(
        lum_fn=source, samples_per_pass=256, grid_rows=12, grid_cols=16)
    original = generator._live()
    assert generator._live() is original
    assert generator.preparation_stats()["hits"] == 1

    source.image = source.image.copy()
    source.image[2:22, 4:28] = 0.6
    source.version = object()
    changed = generator._live()
    assert changed is not original
    assert not np.array_equal(changed[0], original[0])
    assert generator.preparation_stats()["misses"] == 2

    generator.configure(precondition=0.3)
    prepared = generator._live()
    assert prepared is not changed
    assert generator.preparation_stats()["misses"] == 3
    stats = generator.preparation_stats()
    assert stats["entries"] == 1 and stats["bytes"] > 0

    changing = SweepSource(
        lum_fn=lambda: source.image, samples_per_pass=256,
        grid_rows=12, grid_cols=16)
    changing._live()
    changing._live()
    assert changing.preparation_stats()["bypasses"] == 2


def test_sweep_composite_grid_version_key_prevents_mutable_source_staleness():
    versioned = Library()
    generator = SweepSource(samples_per_pass=256, grid_rows=8, grid_cols=12)
    state = {"main": versioned, "mi": 0, "float": None, "fi": 0}
    first = generator._composite(state)
    thumb_reads = versioned.calls
    assert generator._composite(state) is first
    assert versioned.calls == thumb_reads
    assert generator.preparation_stats()["composite_grid_hits"] == 1

    versioned.images[0, :, :, 0] = 20
    versioned.prepared_source_version = object()
    refreshed = generator._composite(state)
    assert refreshed is not first
    assert not np.array_equal(refreshed[0], first[0])

    live = Library(versioned=False)
    live_generator = SweepSource(
        samples_per_pass=256, grid_rows=8, grid_cols=12)
    live_state = {"main": live, "mi": 0, "float": None, "fi": 0}
    live_generator._composite(live_state)
    before = live.calls
    live.images[0, :, :, 0] = 255
    live_generator._composite(live_state)
    assert live.calls == before + 1
    assert live_generator.preparation_stats()["composite_grid_bypasses"] == 2


def test_application_emit_uses_cached_raster_and_stochastic_preparation():
    import scope_display
    from scope_bake import StochasticEmitter

    class Scope:
        samples_per_frame = 640
        samplerate = 48000

        def __init__(self):
            self.frames = []

        def show_frame(self, frame, **_kwargs):
            self.frames.append(np.array(frame, copy=True))
            return frame[-1].copy()

    lib = GeometryLibrary()
    cache = PreparedImageCache()
    scope_display.Scope._tap_until = 0
    for mode in ("raster", "stochastic", "stipple", "vector"):
        reference_scope, cached_scope = Scope(), Scope()
        reference_raster = TraceEmitter(48000, 640, sweep="alternate")
        cached_raster = TraceEmitter(48000, 640, sweep="alternate")
        reference_walk = StochasticEmitter(48000, 640, seed=91)
        cached_walk = StochasticEmitter(48000, 640, seed=91)
        reference_stipple = StippleEmitter(48000, 640, points=48)
        cached_stipple = StippleEmitter(48000, 640, points=48)
        for index in (0, 0, 1, 1):
            args = (lib, None, index, mode, {}, "alternate", 2.2, 0.02,
                    1.0, None, 0.02)
            common = dict(
                emitter=reference_raster,
                stochastic_emitter=reference_walk,
                stipple_emitter=reference_stipple)
            scope_display._emit(reference_scope, *args, **common)
            common.update(
                emitter=cached_raster, stochastic_emitter=cached_walk,
                stipple_emitter=cached_stipple,
                prepared_cache=cache)
            scope_display._emit(cached_scope, *args, **common)
            np.testing.assert_array_equal(
                cached_scope.frames[-1], reference_scope.frames[-1])
            np.testing.assert_array_equal(
                cached_raster._end, reference_raster._end)
            np.testing.assert_array_equal(
                cached_walk._end, reference_walk._end)
            np.testing.assert_array_equal(
                cached_stipple._end, reference_stipple._end)
    stats = cache.snapshot()
    assert stats["hits"] >= 12
    assert "stipple-tour" not in stats["by_kind"]


def test_standalone_field_group_pins_the_source_version_with_its_pixels():
    from scope_frame_scheduler import FieldGroupLatch
    from tools.scope_screen import _begin_raster_field

    class VersionedCapture:
        def __init__(self):
            self.image = np.full((12, 16), 0.25, dtype=np.float32)
            self.version = object()

        def prepared_snapshot(self):
            return self.image, self.version

    capture = VersionedCapture()
    groups = FieldGroupLatch(2)
    levels_for = lambda _image: (None, None)
    field, first = _begin_raster_field(groups, capture, levels_for)
    first_image = first["lum"].copy()
    first_version = first["source_version"]
    assert field == 0
    groups.accept()

    capture.image = np.full((12, 16), 0.75, dtype=np.float32)
    capture.version = object()
    field, second = _begin_raster_field(groups, capture, levels_for)
    assert field == 1 and second["source_version"] is first_version
    np.testing.assert_array_equal(second["lum"], first_image)
    groups.accept()

    field, next_picture = _begin_raster_field(groups, capture, levels_for)
    assert field == 0 and next_picture["source_version"] is capture.version
    assert next_picture["source_version"] is not first_version
    assert np.all(next_picture["lum"] == 0.75)
