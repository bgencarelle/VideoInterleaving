"""Bounded reusable image/geometry preparation, independent of trajectory state."""
from collections import OrderedDict
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from scope_numeric import cumulative_mass, rotate_cloud

from scope_bake import composite_luma

_CACHED_NONE = object()


class PreparedImageCache:
    """Cache prepared images and geometry with an explicit source version.

    Immutable baked libraries supply a unique lifetime token. Mutable sources
    without a version are deliberately recomputed rather than keyed by index.
    Entries own read-only arrays; beam/field/walk/filter state and endpoint-
    dependent stipple tours never live here.
    """

    def __init__(self, max_bytes=8 * 1024 * 1024, max_entries=32):
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max(0, int(max_entries))
        self._entries = OrderedDict()
        self._probation = OrderedDict()
        self._by_kind = {}
        self.bytes = 0
        self.hits = self.misses = self.bypasses = self.evictions = 0

    def _kind_stats(self, kind):
        return self._by_kind.setdefault(
            kind, dict(hits=0, misses=0, bypasses=0, evictions=0,
                       admission_skips=0, admissions=0,
                       probation_evictions=0))

    def _admit_on_miss(self, key, kind, admission):
        if admission <= 1 or self.max_entries <= 0:
            return True
        count = self._probation.pop(key, 0) + 1
        if count >= admission:
            self._kind_stats(kind)["admissions"] += 1
            return True
        self._probation[key] = count
        self._kind_stats(kind)["admission_skips"] += 1
        while len(self._probation) > self.max_entries:
            old_key, _ = self._probation.popitem(last=False)
            self._kind_stats(old_key[0])["probation_evictions"] += 1
        return False

    def _evict_oldest(self):
        key, removed = self._entries.popitem(last=False)
        self.bytes -= self._nbytes(removed)
        self.evictions += 1
        self._kind_stats(key[0])["evictions"] += 1

    def luma(self, main, main_index, floating, float_index, *, bbox=None,
             raw=False, invert=False, rotation=0, thumbnails=None):
        rotation = int(rotation) % 360
        if rotation % 90:
            raise ValueError("prepared-image rotation must be a quarter turn")
        key = self.source_key(
            main, main_index, floating, float_index, bbox=bbox, raw=raw,
            invert=invert, rotation=rotation)

        def build():
            image = composite_luma(main, main_index, floating, float_index,
                                   bbox=bbox, raw=raw, invert=invert,
                                   thumbnails=thumbnails)
            if image is not None and rotation:
                image = np.rot90(image, rotation // 90)
            return image

        return self._prepared("luma", key, (), build)

    @staticmethod
    def source_key(main, main_index, floating, float_index, *, bbox=None,
                   raw=False, invert=False, rotation=0):
        """Return stable provenance for immutable image preparation, else None."""
        sources = (main, floating)
        versions = []
        for lib in sources:
            version = (getattr(lib, "prepared_source_version", None)
                       if lib is not None else None)
            if lib is not None:
                if version is None:
                    return None
                try:
                    hash(version)
                except TypeError:
                    return None
                versions.append((id(lib), version))
            else:
                versions.append((None, None))
        return (tuple(versions), int(main_index), int(float_index),
                tuple(bbox) if bbox is not None else None, bool(raw),
                bool(invert), int(rotation) % 360)

    @staticmethod
    def versioned_array_key(version):
        """Key a standalone live frame only when its producer versions it."""
        if version is None:
            return None
        try:
            hash(version)
        except TypeError:
            return None
        return ("versioned-array", version)

    @staticmethod
    def _freeze(value):
        if isinstance(value, np.ndarray):
            value = np.array(value, copy=True, order="C")
            value.flags.writeable = False
            return value
        if isinstance(value, Mapping):
            return MappingProxyType({key: PreparedImageCache._freeze(item)
                                     for key, item in value.items()})
        if isinstance(value, tuple):
            return tuple(PreparedImageCache._freeze(item) for item in value)
        if isinstance(value, list):
            return tuple(PreparedImageCache._freeze(item) for item in value)
        return value

    @staticmethod
    def _nbytes(value):
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, Mapping):
            return sum(PreparedImageCache._nbytes(item)
                       for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(PreparedImageCache._nbytes(item) for item in value)
        return 0

    def _prepared(self, kind, source_key, settings, builder, *,
                  admit_after=1):
        stage_stats = self._kind_stats(kind)
        if source_key is None:
            self.bypasses += 1
            stage_stats["bypasses"] += 1
            return builder()
        key = (kind, source_key, settings)
        try:
            hash(key)
        except TypeError:
            self.bypasses += 1
            stage_stats["bypasses"] += 1
            return builder()
        if key in self._entries:
            self.hits += 1
            stage_stats["hits"] += 1
            self._entries.move_to_end(key)
            value = self._entries[key]
            return None if value is _CACHED_NONE else value
        self.misses += 1
        stage_stats["misses"] += 1
        value = builder()
        if value is None:
            if self.max_entries:
                while len(self._entries) >= self.max_entries:
                    self._evict_oldest()
                self._entries[key] = _CACHED_NONE
            return None
        size = self._nbytes(value)
        if (not self.max_entries or size > self.max_bytes
                or not self._admit_on_miss(key, kind, int(admit_after))):
            return value
        value = self._freeze(value)
        while self._entries and (len(self._entries) >= self.max_entries
                                 or self.bytes + size > self.max_bytes):
            self._evict_oldest()
        self._entries[key] = value
        self.bytes += size
        return value

    def stochastic_probability(self, luminance, source_key, *, gamma, trim,
                               edge_gain):
        """Reuse immutable stochastic tone/edge probability preparation."""
        from scope_bake import _stochastic_probability
        settings = (float(gamma), float(trim), float(edge_gain))
        return self._prepared(
            "stochastic-probability", source_key, settings,
            lambda: _stochastic_probability(
                luminance, gamma, trim, edge_gain))

    def stochastic_cdf(self, probability, source_key, *, gamma, trim,
                       edge_gain):
        if probability is None:
            return None
        values = np.asarray(probability, dtype=np.float64).ravel()
        settings = (float(gamma), float(trim), float(edge_gain),
                    tuple(np.asarray(probability).shape))
        return self._prepared(
            "stochastic-cdf", source_key, settings,
            lambda: cumulative_mass(values))

    def raster_grid(self, luminance, source_key, *, n, density, trim, rows,
                    cols, autofit, grid_rows, grid_cols, row_bias, levels,
                    stretch, fields, precondition, yt_fixed):
        """Cache immutable raster tone/grid preparation, never a trace path."""
        from scope_bake import prepare_render_grid
        settings = (
            int(n), float(density), rows, cols, float(trim), bool(autofit),
            grid_rows, grid_cols, float(row_bias),
            None if levels is None else tuple(map(float, levels)),
            bool(stretch), int(fields), float(precondition), bool(yt_fixed))
        return self._prepared(
            "raster-grid", source_key, settings,
            lambda: prepare_render_grid(
                luminance, n, density=density, trim=trim, rows=rows,
                cols=cols, autofit=autofit, grid_rows=grid_rows,
                grid_cols=grid_cols, row_bias=row_bias, levels=levels,
                stretch=stretch, fields=fields, precondition=precondition,
                yt_fixed=yt_fixed))

    def vector_frame(self, main, main_index, floating, float_index, *,
                     samples, min_feature=0.02, rotation=0):
        """Cache source-versioned merged/rasterized vector geometry."""
        from scope_bake import merge
        from scope_out import rasterize, rotate_frame
        rotation = int(rotation) % 360
        if rotation % 90:
            raise ValueError("vector geometry rotation must be a quarter turn")
        source_key = self.source_key(
            main, main_index, floating, float_index, rotation=rotation)
        settings = (int(samples), float(min_feature),
                    getattr(main, "flip_y", None),
                    getattr(floating, "flip_y", None))

        def build():
            paths = merge(main, main_index, floating, float_index,
                          min_feature=min_feature)
            frame = rasterize(paths, int(samples))
            return rotate_frame(frame, rotation)

        return self._prepared("vector-frame", source_key, settings, build,
                              admit_after=2)

    def stipple_importance(self, luminance, source_key, *, gamma, trim,
                           edge_gain):
        from scope_bake import _stipple_importance
        settings = (float(gamma), float(trim), float(edge_gain))
        return self._prepared(
            "stipple-importance", source_key, settings,
            lambda: _stipple_importance(
                luminance, gamma, trim, edge_gain))

    def stipple_image_samples(self, importance, source_key, *, points, gamma,
                              trim, edge_gain):
        from scope_bake import _stipple_image_samples
        settings = (int(points), float(gamma), float(trim), float(edge_gain),
                    tuple(np.asarray(importance).shape))
        return self._prepared(
            "stipple-image-samples", source_key, settings,
            lambda: _stipple_image_samples(importance, points))

    def stipple_candidates(self, main, main_index, floating, float_index, *,
                           invert=False, rotation=0):
        from scope_bake import composite_stipple_candidates
        rotation = int(rotation) % 360
        if rotation % 90:
            raise ValueError("stipple rotation must be a quarter turn")
        source_key = self.source_key(
            main, main_index, floating, float_index, raw=True,
            invert=invert, rotation=rotation)

        def build():
            cloud = composite_stipple_candidates(
                main, main_index, floating, float_index, invert=invert)
            if cloud is None or not rotation:
                return cloud
            out = dict(cloud)
            xy = np.asarray(cloud["xy"], dtype=np.float64)
            k = rotation // 90
            out["xy"] = rotate_cloud(xy,k)
            if k % 2:
                out["aspect"] = 1.0 / max(float(cloud.get("aspect", 1.0)), 1e-9)
            return out

        return self._prepared("stipple-candidates", source_key, (), build)

    def stipple_candidate_samples(self, cloud, source_key, *, points, gamma,
                                  trim, edge_gain):
        from scope_bake import _stipple_candidate_samples
        settings = (int(points), float(gamma), float(trim), float(edge_gain))
        return self._prepared(
            "stipple-samples", source_key, settings,
            lambda: _stipple_candidate_samples(
                cloud, points, gamma, trim, edge_gain))

    def stipple_order(self, points, start, source_key, *, settings=()):
        """Memoize a pure tour result; exact accepted start is part of the key."""
        from scope_bake import _greedy_nearest_order
        start_key = tuple(map(float, np.asarray(start).reshape(-1)[:2]))
        points_shape = tuple(np.asarray(points).shape)
        return self._prepared(
            "stipple-tour", source_key,
            (tuple(settings), start_key, points_shape),
            lambda: _greedy_nearest_order(points, start), admit_after=2)

    def snapshot(self):
        by_kind = self._by_kind.copy()
        return dict(hits=self.hits, misses=self.misses, bypasses=self.bypasses,
                    evictions=self.evictions, entries=len(self._entries),
                    bytes=self.bytes, max_bytes=self.max_bytes,
                    max_entries=self.max_entries,
                    by_kind={kind: dict(stats)
                             for kind, stats in by_kind.items()})
