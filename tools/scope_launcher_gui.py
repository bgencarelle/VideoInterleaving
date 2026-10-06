#!/usr/bin/env python3
"""GUI launcher for baked scope playback and arbitrary live/video sources."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scope_out import (list_output_devices, parse_channel_pair,
                       required_output_channels)


RUN_CHOICES = (("VideoInterleaving scope", "app"),
               ("Live screen / video", "live"))
APP_SOURCE_CHOICES = (("Baked XY libraries", "bake"),
                      ("Runtime images", "images"))
RENDER_CHOICES = (("Vector", "vector"), ("Raster", "raster"),
                  ("Stochastic", "stochastic"), ("Stipple", "stipple"),
                  ("Fusion", "fusion"))
IMAGE_RENDER_CHOICES = tuple(item for item in RENDER_CHOICES
                             if item[1] in ("raster", "stochastic", "stipple"))
LIVE_SOURCE_CHOICES = (("Video file / stream", "video"),
                       ("Screen capture · MSS", "screen"),
                       ("Screen / FFmpeg input", "ffmpeg"),
                       ("Camera · FFmpeg input", "camera"),
                       ("Test pattern", "test"))
SWEEP_CHOICES = (("Alternate", "alternate"), ("Palindrome", "palindrome"),
                 ("Retrace", "retrace"))
TRIGGER_SHAPES = (("Ramp · hidden on XY", "ramp"),
                  ("Step · visible dots on XY", "step"))
YT_TIMINGS = (("Dwell", "dwell"), ("Fixed rows", "fixed"))
FUSION_CHOICES = (("Vector + raster + stochastic", "vrs"),
                  ("Vector + raster", "vr"),
                  ("Stochastic + vector", "sv"),
                  ("Stochastic + raster", "sr"))


def window_size_for_workarea(width, height, desired=(1040, 760),
                             horizontal_margin=32, vertical_margin=80):
    """Fit the launcher window inside the monitor's usable work area."""
    width, height = max(1, int(width)), max(1, int(height))
    available_width = max(1, width - int(horizontal_margin))
    available_height = max(1, height - int(vertical_margin))
    return (min(int(desired[0]), available_width),
            min(int(desired[1]), available_height))


def _enumerate_camera_sources():
    """Reuse the modem sender's platform-specific camera discovery."""
    from tools.v7_send_gui import enumerate_camera_sources

    return enumerate_camera_sources()


SELECT_CHOICES = {
    "run_mode": RUN_CHOICES,
    "app_source": APP_SOURCE_CHOICES,
    "render_mode": RENDER_CHOICES,
    "live_source": LIVE_SOURCE_CHOICES,
    "sweep": SWEEP_CHOICES,
    "trigger_shape": TRIGGER_SHAPES,
    "yt_timing": YT_TIMINGS,
    "fusion": FUSION_CHOICES,
}
BOOL_FIELDS = {
    "x_only", "trigger", "scope_gui", "invert", "realtime", "autofit",
    "scope_gui_image_only", "scope_gui_fullscreen",
    "list_from_images", "mirror", "stream",
}
PATH_FIELDS = {"image_dir", "xy_dir", "video_file"}
TEXT_FIELDS = PATH_FIELDS | {"ffmpeg_input", "region", "channels", "display"}


def _settings_defaults():
    import settings as app_settings

    def value(name, fallback):
        return getattr(app_settings, name, fallback)

    return {
        "run_mode": "app",
        "device": "",
        "channels": ",".join(map(str, value("SCOPE_CHANNELS", (1, 2)))),
        "x_only": bool(value("SCOPE_X_ONLY", False)),
        "trigger": bool(value("SCOPE_TRIGGER", True)),
        "trigger_shape": value("SCOPE_TRIGGER_SHAPE", "ramp"),
        "trigger_us": str(value("SCOPE_TRIGGER_US", 250.0)),
        "app_source": value("SCOPE_SOURCE", "bake"),
        "image_dir": str(value("IMAGES_DIR", "images")),
        "xy_dir": str(value("XY_DIR", "") or ""),
        "live_size": str(value("SCOPE_LIVE_SIZE", 128)),
        "render_mode": value("SCOPE_RENDER_MODE", "vector"),
        "invert": bool(value("SCOPE_INVERT", False)),
        "realtime": bool(value("SCOPE_REALTIME", False)),
        "yt_timing": value("SCOPE_YT_TIMING", "dwell"),
        "fields_explicit": bool(value("SCOPE_FIELDS_EXPLICIT", False)),
        "fps": "" if value("SCOPE_FPS", None) is None else str(value("SCOPE_FPS", None)),
        "samples": "" if value("SCOPE_SAMPLES", None) is None else str(value("SCOPE_SAMPLES", None)),
        "geometry_samples": "" if value("SCOPE_GEOMETRY_SAMPLES", None) is None else str(value("SCOPE_GEOMETRY_SAMPLES", None)),
        "traversal_hz": "" if value("SCOPE_TRAVERSAL_HZ", None) is None else str(value("SCOPE_TRAVERSAL_HZ", None)),
        "fields": str(value("SCOPE_FIELDS", 1)),
        "trim": str(value("SCOPE_TRIM", 0.02)),
        "gamma": str(value("SCOPE_GAMMA", 2.2)),
        "density": str(value("SCOPE_DENSITY", 1.0)),
        "precondition": str(value("SCOPE_PRECONDITION", 0.0)),
        "walk_radius": str(value("SCOPE_WALK_RADIUS", 10)),
        "walk_stride": str(value("SCOPE_WALK_STRIDE", 0)),
        "walk_reseed_ms": str(value("SCOPE_WALK_RESEED_MS", 5.0)),
        "stochastic_gamma": str(value("SCOPE_WALK_GAMMA", 2.0)),
        "fusion": value("SCOPE_FUSION", "vrs"),
        "walk_edge": str(value("SCOPE_WALK_EDGE", 0.0)),
        "walk_hz": str(value("SCOPE_WALK_HZ", 48000.0)),
        "stipple_points": str(value("SCOPE_STIPPLE_POINTS", 768)),
        "rows": "" if value("SCOPE_ROWS", None) is None else str(value("SCOPE_ROWS", None)),
        "row_bias": str(value("SCOPE_ROW_BIAS", 1.0)),
        "border": str(value("SCOPE_BORDER", 0.0)),
        "dc_comp": "" if value("SCOPE_DC_COMP", None) is None else str(value("SCOPE_DC_COMP", None)),
        "lowpass": "" if value("SCOPE_LOWPASS", None) is None else str(value("SCOPE_LOWPASS", None)),
        "oversample": str(value("SCOPE_OVERSAMPLE", 1)),
        "autofit": bool(value("SCOPE_AUTOFIT", True)),
        "list_from_images": bool(value("SCOPE_LIST_FROM_IMAGES", False)),
        "mix": "" if value("SCOPE_MIX", None) is None else str(value("SCOPE_MIX", None)),
        "mix_duty": str(value("SCOPE_MIX_DUTY", 0.5)),
        "sweep": value("SCOPE_SWEEP", "alternate"),
        "min_feature": str(value("SCOPE_MIN_FEATURE", 0.02)),
        "rotation": str(value("INITIAL_ROTATION", 0)),
        "mirror": bool(value("INITIAL_MIRROR", False)),
        "scope_gui": bool(value("SCOPE_GUI", False)),
        "scope_gui_image_only": bool(value("SCOPE_GUI_IMAGE_ONLY", False)),
        "scope_gui_fullscreen": bool(value("SCOPE_GUI_FULLSCREEN", False)),
        "live_source": "video",
        "video_file": "",
        "ffmpeg_input": "",
        "display": "",
        "region": "",
        "live_fps": "30",
        "live_samples": "",
        "live_trim": "0.02",
        "live_gamma": "1.8",
        "live_border": "0.0",
        "live_oversample": "1",
        "live_fields": "1",
        "live_density": "1.0",
        "live_rows": "",
        "adapt": "4.0",
        "capture_fps": "12.0",
        "downto": "160",
        "stream": False,
        "buffer_blocks": "6",
        "live_dc_comp": "",
        "blocksize": "1024",
    }


DEFAULT_SETTINGS = _settings_defaults()
if (DEFAULT_SETTINGS["app_source"] == "images" and
        DEFAULT_SETTINGS["render_mode"] not in dict(IMAGE_RENDER_CHOICES).values()):
    DEFAULT_SETTINGS["render_mode"] = "raster"

FIELD_GROUPS = (
    ("Pipeline", ("run_mode", "device")),
    ("Output and trigger", ("channels", "x_only", "trigger",
                             "trigger_shape", "trigger_us", "lowpass",
                             "rotation", "mirror")),
    ("Image source", ("app_source", "image_dir", "xy_dir", "live_size")),
    ("Renderer and timing", ("render_mode", "fps", "samples",
                              "geometry_samples", "traversal_hz", "fields",
                              "yt_timing", "realtime", "sweep")),
    ("Raster and tone", ("invert", "trim", "gamma", "density", "rows",
                          "row_bias", "border", "precondition", "autofit")),
    ("Stochastic, stipple, fusion", ("walk_radius", "walk_stride",
                                     "walk_reseed_ms", "stochastic_gamma",
                                     "walk_edge", "walk_hz", "stipple_points",
                                     "fusion")),
    ("Output processing", ("dc_comp", "oversample", "mix", "mix_duty",
                            "min_feature", "list_from_images", "scope_gui",
                            "scope_gui_image_only", "scope_gui_fullscreen")),
    ("Live source", ("live_source", "video_file", "ffmpeg_input", "display",
                      "region")),
    ("Live sweep and capture", ("live_fps", "live_samples",
                                  "geometry_samples", "traversal_hz", "live_trim",
                                 "live_gamma", "live_density", "live_rows",
                                 "live_fields", "live_border", "live_oversample",
                                 "adapt",
                                 "capture_fps", "downto", "stream",
                                 "buffer_blocks", "live_dc_comp", "blocksize")),
    ("Live visualizer", ("scope_gui",)),
)

FIELD_LABELS = {
    "run_mode": "Scope pipeline", "device": "Audio output device",
    "channels": "PortAudio channels · X,Y", "x_only": "X-only mono · Y-T",
    "trigger": "X trigger marker", "trigger_shape": "Trigger shape",
    "trigger_us": "Trigger duration · microseconds", "lowpass": "Output low-pass · Hz",
    "app_source": "Scope input", "image_dir": "Image source folder",
    "xy_dir": "Baked XY folder", "live_size": "Runtime thumbnail width",
    "render_mode": "Renderer", "fps": "Scope traces / second",
    "samples": "Samples per trace · overrides FPS", "fields": "Raster fields",
    "geometry_samples": "Geometry samples", "traversal_hz": "Traversal Hz",
    "yt_timing": "Raster row timing", "realtime": "Realtime raster stream",
    "sweep": "Raster/vector sweep", "invert": "Invert image luminance",
    "trim": "Trim cutoff", "gamma": "Raster gamma", "density": "Samples / cell",
    "rows": "Raster rows · blank = auto", "row_bias": "Raster row bias",
    "border": "Fixed border fraction", "precondition": "Raster precondition",
    "autofit": "Autofit trimmed raster grid", "walk_radius": "Walk radius · px",
    "walk_stride": "Walk stride · 0 = automatic", "walk_reseed_ms": "Walk reseed · ms",
    "stochastic_gamma": "Stochastic gamma", "walk_edge": "Walk edge probability",
    "walk_hz": "Stochastic target clock · Hz", "stipple_points": "Stipple points",
    "fusion": "Fusion components", "dc_comp": "DC compensation · Hz",
    "oversample": "Anti-alias oversampling", "mix": "Whole-trace mix · Hz, blank off",
    "mix_duty": "Mix raster duty", "min_feature": "Vector minimum feature",
    "rotation": "Rotation · degrees", "mirror": "Mirror X",
    "list_from_images": "Rebuild legacy image manifest",
    "scope_gui": "Open native live tuner",
    "scope_gui_image_only": "Start tuner in image-only view",
    "scope_gui_fullscreen": "Start fullscreen · image-only",
    "live_source": "Live source", "video_file": "Video file / URL",
    "ffmpeg_input": "FFmpeg input · FMT:SRC", "display": "Capture display index",
    "region": "Capture region · left,top,width,height", "live_fps": "Trace rate · Hz",
    "live_samples": "Samples / trace · overrides rate", "live_trim": "Trim cutoff",
    "live_gamma": "Tone gamma", "live_density": "Samples / cell",
    "live_rows": "Scanlines · blank = auto", "live_fields": "Interlace fields",
    "live_border": "Fixed border fraction", "live_oversample": "Anti-alias oversampling",
    "adapt": "Tone adaptation · seconds, 0 off",
    "capture_fps": "Capture rate · frames / second", "downto": "Capture width",
    "stream": "Low-latency streaming renderer", "buffer_blocks": "Audio buffer blocks",
    "live_dc_comp": "DC compensation · Hz", "blocksize": "Audio callback block size",
}

FILE_FIELDS = {"video_file"}
DIR_FIELDS = {"image_dir", "xy_dir"}
NUMBER_FIELDS = {
    "trigger_us", "live_size", "fps", "samples", "geometry_samples",
    "traversal_hz", "fields", "trim", "gamma",
    "density", "precondition", "walk_radius", "walk_stride", "walk_reseed_ms",
    "stochastic_gamma", "walk_edge", "walk_hz", "stipple_points", "rows",
    "row_bias", "border", "dc_comp", "lowpass", "oversample", "mix",
    "mix_duty", "min_feature", "rotation", "live_fps", "live_samples",
    "live_trim", "live_gamma", "live_border", "live_oversample", "live_fields",
    "live_density", "live_rows", "adapt", "capture_fps", "downto",
    "buffer_blocks", "live_dc_comp", "blocksize",
}


def _number(settings, key, *, optional=False, integer=False):
    value = str(settings.get(key, "")).strip()
    if not value and optional:
        return None
    try:
        result = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{FIELD_LABELS.get(key, key)} must be a number") from exc
    if not math.isfinite(float(result)):
        raise ValueError(f"{FIELD_LABELS.get(key, key)} must be finite")
    return result


def _parse_region(value):
    if not str(value or "").strip():
        return None
    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except (AttributeError, ValueError) as exc:
        raise ValueError("Capture region must be left,top,width,height") from exc
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError("Capture region must be left,top,width,height with positive size")
    return parts


def validate_settings(settings, outputs=(), root=ROOT):
    """Validate the selected pipeline and its source-specific settings."""
    run_mode = settings.get("run_mode", "app")
    if run_mode not in ("app", "live"):
        raise ValueError("Choose a scope pipeline")
    channels = parse_channel_pair(settings.get("channels", "1,2"))
    x_only = bool(settings.get("x_only", False))
    min_channels = required_output_channels(channels, x_only)
    device = str(settings.get("device", "")).strip()
    if device and device.lower() != "null":
        try:
            device_index = int(device)
        except ValueError as exc:
            raise ValueError("Choose an audio output from the device list") from exc
        if outputs and not 0 <= device_index < len(outputs):
            raise ValueError("The selected device does not have enough output channels")
    lowpass = _number(settings, "lowpass", optional=True)
    if lowpass is not None and lowpass <= 0:
        raise ValueError("Output low-pass must be greater than zero")
    if _number(settings, "trigger_us") <= 0:
        raise ValueError("Trigger duration must be greater than zero")
    rotation = _number(settings, "rotation", integer=True)
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Rotation must be 0, 90, 180, or 270 degrees")

    if run_mode == "app":
        source = settings.get("app_source", "bake")
        renderer = settings.get("render_mode", "vector")
        if source not in ("bake", "images"):
            raise ValueError("Choose baked XY or runtime images")
        if renderer not in dict(RENDER_CHOICES).values():
            raise ValueError("Choose a supported scope renderer")
        if source == "images" and renderer not in dict(IMAGE_RENDER_CHOICES).values():
            raise ValueError("Runtime images support raster, stochastic, or stipple")
        if settings.get("trigger_shape", "ramp") not in dict(TRIGGER_SHAPES).values():
            raise ValueError("Choose a supported trigger shape")
        if settings.get("yt_timing", "dwell") not in dict(YT_TIMINGS).values():
            raise ValueError("Choose a supported raster row timing")
        geometry_samples = _number(settings, "geometry_samples", optional=True,
                                   integer=True)
        traversal_hz = _number(settings, "traversal_hz", optional=True)
        if geometry_samples is not None or traversal_hz is not None:
            if source != "bake" or renderer != "raster":
                raise ValueError("Geometry/traversal controls require baked raster rendering")
            if settings.get("yt_timing", "dwell") == "fixed":
                raise ValueError("Independent geometry controls cannot use fixed Y-T timing")
            if _number(settings, "mix", optional=True) is not None:
                raise ValueError("Geometry/traversal controls cannot be combined with whole-trace mix")
            if geometry_samples is not None and geometry_samples < 2:
                raise ValueError("Geometry samples must be at least 2")
            if traversal_hz is not None and traversal_hz <= 0:
                raise ValueError("Traversal Hz must be greater than zero")
            if traversal_hz is not None and not settings.get("trigger", True):
                raise ValueError("Traversal Hz requires the scope trigger")
            if traversal_hz is not None and _number(settings, "fields", integer=True) > 1:
                raise ValueError("Traversal Hz requires one field")
            if settings.get("realtime"):
                raise ValueError("Geometry/traversal controls cannot be used in realtime mode")
        if settings.get("sweep", "alternate") not in dict(SWEEP_CHOICES).values():
            raise ValueError("Choose a supported raster/vector sweep")
        mix_enabled = source == "bake" and renderer != "fusion"
        mix_hz = (_number(settings, "mix", optional=True)
                  if mix_enabled else None)
        if mix_hz is not None and mix_hz <= 0:
            raise ValueError("Whole-trace mix rate must be greater than zero")
        if mix_hz is not None and _number(settings, "samples", optional=True) is not None:
            raise ValueError("Whole-trace mix cannot be combined with samples per trace")
        mix_duty = _number(settings, "mix_duty") if mix_enabled else 0.5
        if mix_enabled and not 0.0 <= mix_duty <= 1.0:
            raise ValueError("Mix raster duty must be between 0 and 1")
        if source == "images":
            image_dir = str(settings.get("image_dir", "")).strip()
            if image_dir and not (Path(image_dir).expanduser().is_absolute()
                                  and Path(image_dir).expanduser().is_dir()) and not (
                    Path(root, image_dir).is_dir()):
                raise ValueError(f"Image source folder not found: {image_dir}")
        for key in ("fps", "samples", "live_size", "fields", "walk_radius",
                    "walk_stride", "stipple_points", "rows", "oversample"):
            value = _number(settings, key, optional=True, integer=True)
            minimum = 0 if key == "walk_stride" else 1
            if value is not None and value < minimum:
                qualifier = "non-negative" if minimum == 0 else "greater than zero"
                raise ValueError(f"{FIELD_LABELS[key]} must be {qualifier}")
        if _number(settings, "stipple_points", integer=True) < 8:
            raise ValueError("Stipple points must be at least 8")
        for key in ("trim", "gamma", "density", "precondition",
                    "walk_stride", "walk_reseed_ms", "stochastic_gamma",
                    "walk_edge", "walk_hz", "stipple_points", "rows",
                    "row_bias", "border", "dc_comp", "lowpass", "min_feature"):
            value = _number(settings, key, optional=True)
            if value is not None and value < 0:
                raise ValueError(f"{FIELD_LABELS[key]} cannot be negative")
        for key in ("gamma", "density", "walk_hz"):
            if _number(settings, key) <= 0:
                raise ValueError(f"{FIELD_LABELS[key]} must be greater than zero")
        for key in ("trim", "border", "walk_edge"):
            value = _number(settings, key)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{FIELD_LABELS[key]} must be between 0 and 1")
        if source == "bake":
            xy_dir = str(settings.get("xy_dir", "")).strip()
            if (xy_dir and not Path(xy_dir).expanduser().is_dir()
                    and not Path(root, xy_dir).is_dir()):
                raise ValueError(f"Baked XY folder not found: {xy_dir}")
        return {"run_mode": run_mode, "channels": channels,
                "min_channels": min_channels, "device": device}

    source = settings.get("live_source", "test")
    if source not in dict(LIVE_SOURCE_CHOICES).values():
        raise ValueError("Choose a supported live source")
    if settings.get("trigger_shape", "ramp") not in dict(TRIGGER_SHAPES).values():
        raise ValueError("Choose a supported trigger shape")
    if source == "video" and not str(settings.get("video_file", "")).strip():
        raise ValueError("Choose a video file or URL")
    geometry_samples = _number(settings, "geometry_samples", optional=True,
                               integer=True)
    traversal_hz = _number(settings, "traversal_hz", optional=True)
    if geometry_samples is not None and geometry_samples < 2:
        raise ValueError("Geometry samples must be at least 2")
    if traversal_hz is not None:
        if traversal_hz <= 0:
            raise ValueError("Traversal Hz must be greater than zero")
        if not settings.get("trigger", True):
            raise ValueError("Traversal Hz requires the scope trigger")
        if _number(settings, "live_fields", integer=True) > 1:
            raise ValueError("Traversal Hz requires one field")
    if (geometry_samples is not None or traversal_hz is not None) and settings.get("stream"):
        raise ValueError("Geometry/traversal controls require whole-trace rendering")
    if source == "camera" and not str(settings.get("ffmpeg_input", "")).strip():
        raise ValueError("Choose a camera device or FFmpeg input, e.g. v4l2:/dev/video0")
    if source in ("camera", "ffmpeg"):
        input_spec = str(settings.get("ffmpeg_input", "")).strip()
        if input_spec and ":" not in input_spec:
            raise ValueError("FFmpeg input must use FMT:SRC syntax")
    _parse_region(settings.get("region", ""))
    if source == "video":
        path = str(settings.get("video_file", "")).strip()
        parsed_url = urlparse(path)
        if not (parsed_url.scheme and parsed_url.netloc):
            if not Path(path).expanduser().is_file() and not Path(root, path).is_file():
                raise ValueError(f"Video file not found: {path}")
    for key in ("live_fps", "live_samples", "live_fields", "downto",
                "buffer_blocks", "live_oversample", "live_rows"):
        value = _number(settings, key, optional=True, integer=True)
        if value is not None and value <= 0:
            raise ValueError(f"{FIELD_LABELS[key]} must be greater than zero")
    for key in ("live_trim", "live_gamma", "live_density",
                "live_border", "adapt", "capture_fps", "live_dc_comp",
                "lowpass"):
        value = _number(settings, key, optional=True)
        if value is not None and value < 0:
            raise ValueError(f"{FIELD_LABELS[key]} cannot be negative")
    if _number(settings, "blocksize", integer=True) < 0:
        raise ValueError("Audio callback block size cannot be negative")
    if _number(settings, "live_gamma") <= 0:
        raise ValueError("Tone gamma must be greater than zero")
    if _number(settings, "live_density") <= 0:
        raise ValueError("Samples per cell must be greater than zero")
    if _number(settings, "capture_fps") <= 0:
        raise ValueError("Capture rate must be greater than zero")
    live_trim = _number(settings, "live_trim")
    if not 0.0 <= live_trim <= 1.0:
        raise ValueError("Trim cutoff must be between 0 and 1")
    live_border = _number(settings, "live_border")
    if not 0.0 <= live_border <= 1.0:
        raise ValueError("Fixed border fraction must be between 0 and 1")
    start = settings.get("start_at", 0)
    if not math.isfinite(float(start)) or float(start) < 0:
        raise ValueError("Video resume position must be non-negative")
    return {"run_mode": run_mode, "channels": channels,
            "min_channels": min_channels, "device": device}


def build_command(settings, outputs=(), *, python=sys.executable, root=ROOT,
                  video_start=None):
    """Build an argv list for either scope pipeline without invoking a shell."""
    checked = validate_settings(settings, outputs, root)
    command = [str(python)]
    if settings.get("run_mode", "app") == "app":
        command.extend((str(Path(root) / "main.py"), "--mode", "scope"))
        for key, flag in (("image_dir", "--dir"), ("xy_dir", "--xy-dir")):
            value = str(settings.get(key, "")).strip()
            if value:
                command.extend((flag, str(Path(value).expanduser())))
        command.extend(("--scope-source", settings.get("app_source", "bake")))
        command.extend(("--scope-mode", settings.get("render_mode", "vector")))
        command.extend(("--scope-channels", ",".join(map(str, checked["channels"]))))
        if checked["device"]:
            command.extend(("--device", checked["device"]))
        command.extend(("--scope-x-only" if settings.get("x_only")
                        else "--no-scope-x-only",))
        command.extend(("--scope-trigger" if settings.get("trigger")
                        else "--no-scope-trigger",))
        command.extend(("--scope-trigger-shape", settings.get("trigger_shape", "ramp")))
        command.extend(("--scope-trigger-us", str(_number(settings, "trigger_us"))))
        command.extend(("--scope-yt-timing", settings.get("yt_timing", "dwell")))
        if not str(settings.get("fps", "")).strip():
            command.append("--no-scope-fps")
        if not str(settings.get("samples", "")).strip():
            command.append("--no-scope-samples")
        if not str(settings.get("rows", "")).strip():
            command.append("--no-scope-rows")
        if not str(settings.get("dc_comp", "")).strip():
            command.append("--no-scope-dc-comp")
        if not str(settings.get("lowpass", "")).strip():
            command.append("--no-scope-lowpass")
        mix_enabled = (settings.get("app_source", "bake") == "bake" and
                       settings.get("render_mode") != "fusion")
        if not mix_enabled or not str(settings.get("mix", "")).strip():
            command.append("--no-scope-mix")
        for key, flag in (("live_size", "--scope-live-size"),
                          ("fps", "--scope-fps"), ("samples", "--scope-samples"),
                          ("geometry_samples", "--scope-geometry-samples"),
                          ("traversal_hz", "--scope-traversal-hz"),
                          ("fields", "--scope-fields"), ("trim", "--scope-trim"),
                          ("gamma", "--scope-gamma"), ("density", "--scope-density"),
                          ("precondition", "--scope-precondition"),
                          ("walk_radius", "--scope-walk-radius"),
                          ("walk_stride", "--scope-walk-stride"),
                          ("walk_reseed_ms", "--scope-walk-reseed-ms"),
                          ("stochastic_gamma", "--scope-stochastic-gamma"),
                          ("fusion", "--scope-fusion"),
                          ("walk_edge", "--scope-walk-edge"),
                          ("walk_hz", "--scope-walk-hz"),
                          ("stipple_points", "--scope-stipple-points"),
                          ("rows", "--scope-rows"),
                          ("row_bias", "--scope-row-bias"),
                          ("border", "--scope-border"),
                          ("dc_comp", "--scope-dc-comp"),
                          ("lowpass", "--scope-lowpass"),
                          ("oversample", "--scope-oversample"),
                          ("mix", "--scope-mix"),
                          ("mix_duty", "--scope-mix-duty"),
                          ("sweep", "--scope-sweep"),
                          ("min_feature", "--scope-min-feature"),
                          ("rotation", "--rotation")):
            if key in ("mix", "mix_duty") and not mix_enabled:
                continue
            value = str(settings.get(key, "")).strip()
            if not value:
                continue
            if (key == "fields" and value == str(DEFAULT_SETTINGS["fields"])
                    and not settings.get("fields_explicit", False)):
                # An explicit --scope-fields 1 changes mix's auto-field rule.
                continue
            command.extend((flag, value))
        for key, flag in (("invert", "--scope-invert"),
                          ("mirror", "--mirror")):
            command.append(flag if settings.get(key) else
                           "--no-" + flag[2:])
        command.append("--scope-realtime" if settings.get("realtime")
                       else "--no-scope-realtime")
        command.append("--scope-autofit" if settings.get("autofit", True)
                       else "--scope-no-autofit")
        use_image_manifest = (settings.get("list_from_images") and
                              settings.get("app_source", "bake") == "bake")
        command.append("--scope-list-from-images" if use_image_manifest
                       else "--no-scope-list-from-images")
        start_image_only = bool(settings.get("scope_gui_image_only") or
                                settings.get("scope_gui_fullscreen"))
        start_fullscreen = bool(settings.get("scope_gui_fullscreen"))
        gui_enabled = bool(settings.get("scope_gui") or start_image_only
                           or start_fullscreen)
        command.append("--scope-gui" if gui_enabled else "--no-scope-gui")
        if start_image_only:
            command.append("--scope-gui-image-only")
        if start_fullscreen:
            command.append("--scope-gui-fullscreen")
        return command

    command.extend((str(Path(root) / "tools" / "scope_screen.py"),
                    "--source", settings.get("live_source", "test")))
    if checked["device"]:
        command.extend(("--device", checked["device"]))
    command.extend(("--scope-channels", ",".join(map(str, checked["channels"]))))
    command.extend(("--scope-x-only" if settings.get("x_only")
                    else "--no-scope-x-only",))
    command.extend(("--scope-trigger" if settings.get("trigger")
                    else "--no-scope-trigger",))
    command.extend(("--scope-trigger-shape", settings.get("trigger_shape", "ramp"),
                    "--scope-trigger-us", str(_number(settings, "trigger_us"))))
    command.extend(("--rotation", str(_number(settings, "rotation", integer=True))))
    command.append("--mirror" if settings.get("mirror") else "--no-mirror")
    if settings.get("live_source") == "video":
        video_path = str(settings["video_file"]).strip()
        parsed_video = urlparse(video_path)
        is_stream = bool(parsed_video.scheme and parsed_video.netloc)
        if not is_stream:
            video_path = str(Path(video_path).expanduser())
        command.extend(("--file", video_path, "--control"))
        start = video_start
        if start is None:
            start = settings.get("start_at", 0.0)
        try:
            start = float(start or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("Video resume position must be a number") from exc
        if not math.isfinite(start) or start < 0:
            raise ValueError("Video resume position must be finite and non-negative")
        if start > 0 and not is_stream:
            command.extend(("--start-at", str(start)))
    if settings.get("live_source") in ("camera", "ffmpeg"):
        spec = str(settings.get("ffmpeg_input", "")).strip()
        if spec:
            command.extend(("--ffmpeg-input", spec))
    if str(settings.get("display", "")).strip():
        command.extend(("--display", str(settings["display"]).strip()))
    if str(settings.get("region", "")).strip():
        command.extend(("--region", str(settings["region"]).strip()))
    for key, flag in (("live_fps", "--fps"), ("live_samples", "--samples"),
                      ("geometry_samples", "--geometry-samples"),
                      ("traversal_hz", "--traversal-hz"),
                      ("live_trim", "--trim"), ("live_gamma", "--gamma"),
                      ("live_border", "--border"),
                      ("live_oversample", "--oversample"),
                      ("live_fields", "--fields"), ("live_density", "--density"),
                      ("live_rows", "--rows"), ("adapt", "--adapt"),
                      ("capture_fps", "--capture-fps"), ("downto", "--downto"),
                      ("buffer_blocks", "--buffer-blocks"),
                      ("live_dc_comp", "--dc-comp"), ("blocksize", "--blocksize"),
                      ("lowpass", "--scope-lowpass")):
        value = str(settings.get(key, "")).strip()
        if value:
            command.extend((flag, value))
    if settings.get("stream"):
        command.append("--stream")
    if settings.get("scope_gui"):
        command.append("--scope-gui")
    return command


def preferences_path():
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "video-interleaving" / "scope-gui.json"


def load_preferences(path):
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}, {}
    if not isinstance(document, dict) or document.get("version") != 1:
        return {}, {}
    settings = document.get("settings")
    resume = document.get("resume")
    return (settings if isinstance(settings, dict) else {},
            resume if isinstance(resume, dict) else {})


def save_preferences(path, settings, resume):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"version": 1, "settings": settings,
                                     "resume": resume}, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _file_picker(current="", directory=False):
    """Use a native host file chooser without making another GUI dependency."""
    import platform

    current_path = Path(str(current)).expanduser() if current else None
    start = (current_path if current_path and current_path.is_dir() else
             current_path.parent if current_path and current_path.parent.is_dir()
             else ROOT)
    system = platform.system().lower()
    if system == "darwin":
        executable = shutil.which("osascript")
        if executable is None:
            raise RuntimeError("macOS file picker (osascript) is unavailable")
        command = [executable, "-e", ("POSIX path of (choose folder)" if directory
                                      else "POSIX path of (choose file)")]
    elif system == "windows":
        executable = shutil.which("powershell.exe") or shutil.which("powershell")
        if executable is None:
            raise RuntimeError("Windows PowerShell file picker is unavailable")
        if directory:
            script = ("Add-Type -AssemblyName System.Windows.Forms; "
                      "$d = New-Object System.Windows.Forms.FolderBrowserDialog; "
                      f"$d.SelectedPath = '{str(start).replace(chr(39), chr(39)*2)}'; "
                      "if ($d.ShowDialog() -eq 'OK') { [Console]::Write($d.SelectedPath) }")
        else:
            script = ("Add-Type -AssemblyName System.Windows.Forms; "
                      "$d = New-Object System.Windows.Forms.OpenFileDialog; "
                      f"$d.InitialDirectory = '{str(start).replace(chr(39), chr(39)*2)}'; "
                      "$d.Filter = 'Video files|*.avi;*.mkv;*.mov;*.mp4;*.webm|All files|*.*'; "
                      "if ($d.ShowDialog() -eq 'OK') { [Console]::Write($d.FileName) }")
        command = [executable, "-NoProfile", "-STA", "-Command", script]
    else:
        executable = shutil.which("zenity")
        if executable:
            command = ([executable, "--file-selection", "--directory",
                        f"--filename={str(start)}/"] if directory else
                       [executable, "--file-selection", f"--filename={str(start)}/",
                        "--file-filter=Video files | *.avi *.mkv *.mov *.mp4 *.webm"])
        else:
            executable = shutil.which("kdialog")
            if executable is None:
                raise RuntimeError("Install zenity or kdialog, or type/drop the path")
            command = ([executable, "--getexistingdirectory", str(start)] if directory
                       else [executable, "--getopenfilename", str(start),
                             "Video files (*.avi *.mkv *.mov *.mp4 *.webm)"])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


class ScopeLauncher:
    WINDOW_SIZE = (1040, 760)
    ROW_HEIGHT = 31
    FOOTER_HEIGHT = 42
    TOOLBAR_HEIGHT = 56

    def __init__(self, device_error="", preference_path=None,
                 restore_preferences=True):
        self.device_error = device_error
        self._device_cache = {}
        self._camera_cache = None
        self.preference_path = Path(preference_path or preferences_path())
        self.settings = dict(DEFAULT_SETTINGS)
        self.resume = {}
        if restore_preferences:
            saved, self.resume = load_preferences(self.preference_path)
            self.settings.update({key: value for key, value in saved.items()
                                  if key in self.settings})
        if (self.settings.get("app_source") == "images" and
                self.settings.get("render_mode") not in
                dict(IMAGE_RENDER_CHOICES).values()):
            self.settings["render_mode"] = "raster"
        self.page = "setup"
        self.selected = "run_mode"
        self.scroll = 0
        self.dropdown = None
        self.editing = False
        self.edit_buffer = ""
        self.notice = device_error or "Choose a pipeline, source, and output; then Start."
        self.process = None
        self.reader = None
        self.events = queue.Queue()
        self.lines = []
        self.playback = None
        self.seek_drag = None
        self.seek_preview = None
        self.active_video_transport = False
        self.stop_requested = False
        self.close_when_stopped = False
        self._last_resume_save = 0.0
        self.dirty = True
        self.hits = {}
        self._height = self.WINDOW_SIZE[1]
        self._ui_scale = 1.0
        self._windowed_bounds = None
        self.work_area = None
        self.primary_monitor = None
        self._last_sizes = None
        self.frame = None
        self.texture = None
        self.context = None
        self.program = None
        self.vao = None
        self.fullscreen = False
        self.closed = False
        self._init_graphics()

    def _init_graphics(self):
        try:
            import glfw
            import moderngl
            from PIL import Image, ImageDraw, ImageFont
        except Exception as exc:
            raise RuntimeError("Scope GUI needs glfw, moderngl, and Pillow: "
                               + str(exc)) from exc
        self.glfw, self.moderngl = glfw, moderngl
        self.Image, self.ImageDraw, self.ImageFont = Image, ImageDraw, ImageFont
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed for scope GUI")
        monitor = glfw.get_primary_monitor()
        work_area = None
        if monitor is not None:
            get_workarea = getattr(glfw, "get_monitor_workarea", None)
            try:
                if get_workarea is not None:
                    work_area = tuple(int(value) for value in
                                      get_workarea(monitor))
            except Exception:
                work_area = None
            if work_area is None:
                try:
                    mode = glfw.get_video_mode(monitor)
                    monitor_x, monitor_y = glfw.get_monitor_pos(monitor)
                    work_area = (int(monitor_x), int(monitor_y),
                                 int(mode.size.width), int(mode.size.height))
                except Exception:
                    work_area = None
        self.primary_monitor = monitor
        self.work_area = work_area
        if work_area is not None and len(work_area) == 4:
            area_x, area_y, area_width, area_height = work_area
            window_size = window_size_for_workarea(
                area_width, area_height, self.WINDOW_SIZE)
            min_width = min(720, max(1, area_width - 32))
            min_height = min(520, max(1, area_height - 80))
        else:
            window_size = self.WINDOW_SIZE
            min_width, min_height = 720, 520
        failures = []
        for use_gles in (False, True):
            glfw.default_window_hints()
            if use_gles:
                glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
                glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
                version = 300
                shader_version = "#version 300 es\nprecision highp float;\n"
            else:
                glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
                glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
                glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
                glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
                version = 330
                shader_version = "#version 330\n"
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
            window = glfw.create_window(*window_size,
                                        "VideoInterleaving · Scope", None, None)
            if not window:
                failures.append("GLES" if use_gles else "OpenGL")
                continue
            context = None
            try:
                glfw.make_context_current(window)
                context = moderngl.create_context(require=version)
                if getattr(context, "version_code", 0) < 300:
                    raise RuntimeError("OpenGL ES 3.0 / OpenGL 3.3 is required")
                self.window = window
                self.context = context
                self._shader_version = shader_version
                break
            except Exception as exc:
                failures.append(str(exc))
                if context is not None:
                    try:
                        context.release()
                    except Exception:
                        pass
                glfw.make_context_current(None)
                glfw.destroy_window(window)
        else:
            glfw.terminate()
            self.window = None
            raise RuntimeError("Could not create scope GUI graphics context: "
                               + "; ".join(failures))
        self.program = self.context.program(
            vertex_shader=self._shader_version + "out vec2 uv; void main(){"
            "vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);"
            "// The visible part interpolates p over [0,1], not [0,2].\n"
            "uv=vec2(p.x,1.-p.y);gl_Position=vec4(p*2.-1.,0.,1.);}",
            fragment_shader=self._shader_version + "uniform sampler2D tex;in vec2 uv;"
            "out vec4 color;void main(){color=texture(tex,uv);}")
        self.program["tex"].value = 0
        self.vao = self.context.vertex_array(self.program, [])
        self.font = self._font(17)
        self.small = self._font(13)
        self.work_area = work_area
        glfw.set_window_close_callback(self.window,
                                       lambda _w: setattr(self, "close_when_stopped", True))
        glfw.set_window_size_limits(self.window, min_width, min_height,
                                    glfw.DONT_CARE, glfw.DONT_CARE)
        if work_area is not None:
            try:
                glfw.set_window_pos(
                    self.window,
                    area_x + max(0, (area_width - window_size[0]) // 2),
                    area_y + max(0, (area_height - window_size[1]) // 2))
            except Exception:
                pass
        glfw.set_mouse_button_callback(self.window, self._on_mouse)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_char_callback(self.window, self._on_char)
        glfw.set_drop_callback(self.window, self._on_drop)
        glfw.set_window_size_callback(self.window, self._on_resize)
        glfw.set_framebuffer_size_callback(self.window, self._on_resize)

    def _on_resize(self, _window, *_size):
        """A configure or framebuffer change requires a fresh canvas upload."""
        self.dirty = True

    def _apply_ui_scale(self, height):
        scale = min(1.35, max(0.75, int(height) / self.WINDOW_SIZE[1]))
        if scale == self._ui_scale:
            return
        self._ui_scale = scale
        self.ROW_HEIGHT = max(24, round(31 * scale))
        self.FOOTER_HEIGHT = max(34, round(42 * scale))
        self.TOOLBAR_HEIGHT = max(44, round(56 * scale))
        self.font = self._font(max(12, round(17 * scale)))
        self.small = self._font(max(10, round(13 * scale)))

    def _u(self, value):
        return max(1, round(value * self._ui_scale))

    def _font(self, size):
        try:
            return self.ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            try:
                return self.ImageFont.load_default(size=size)
            except TypeError:
                return self.ImageFont.load_default()

    def _min_channels(self):
        try:
            pair = parse_channel_pair(self.settings.get("channels", "1,2"))
            return required_output_channels(pair, self.settings.get("x_only", False))
        except ValueError:
            return 1 if self.settings.get("x_only") else 2

    def _device_choices(self, refresh=False):
        min_channels = self._min_channels()
        if refresh or min_channels not in self._device_cache:
            try:
                devices = tuple(list_output_devices(min_channels=min_channels))
                error = ("" if devices else
                         f"No audio output has at least {min_channels} channel(s).")
                self._device_cache[min_channels] = (devices, error)
            except Exception as exc:
                error = f"Audio output enumeration failed: {exc}"
                self._device_cache[min_channels] = ((), error)
        devices, error = self._device_cache[min_channels]
        if error:
            self.device_error = error
            self.notice = error
        choices = [("System default / automatic", "")]
        choices.append(("Null · virtual scope output", "null"))
        choices.extend((f"[{index}] {name} · {api} · {rate or '?'} Hz", str(index))
                       for index, (_global, name, api, rate) in enumerate(devices))
        return tuple(choices), devices

    def _camera_choices(self, refresh=False):
        if refresh or self._camera_cache is None:
            try:
                choices = tuple(_enumerate_camera_sources())
                error = "" if choices else "No camera devices were found."
            except Exception as exc:
                choices, error = (), str(exc)
            self._camera_cache = (choices, error)
        return self._camera_cache

    def _camera_notice(self):
        _choices, error = self._camera_choices()
        if "No camera devices" in error:
            return ("No camera devices found. Check connection and permissions, "
                    "or use Screen / FFmpeg input.")
        if "FFmpeg is required" in error:
            return ("Camera discovery needs FFmpeg on this platform. Install "
                    "FFmpeg, or use Screen / FFmpeg input.")
        return (f"Video device enumeration failed: {error}. "
                "Try Screen / FFmpeg input.")

    def _is_dropdown_field(self, key):
        return (key in SELECT_CHOICES or key == "device" or
                (key == "ffmpeg_input" and
                 self.settings.get("live_source") == "camera"))

    def _open_dropdown(self, key):
        if self._is_dropdown_field(key) and not self._choices(key):
            self.dropdown = None
            camera_field = (key == "ffmpeg_input" and
                            self.settings.get("live_source") == "camera")
            if camera_field:
                self.notice = self._camera_notice()
            else:
                self.notice = "No choices are available for this setting."
            self.dirty = True
            return
        self.dropdown = key

    def _choices(self, key):
        if key == "device":
            return self._device_choices()[0]
        if key == "ffmpeg_input" and self.settings.get("live_source") == "camera":
            return self._camera_choices()[0]
        if key == "render_mode" and self.settings.get("app_source") == "images":
            return IMAGE_RENDER_CHOICES
        return SELECT_CHOICES.get(key, ())

    def _sections(self):
        groups = []
        run_mode = self.settings.get("run_mode", "app")
        for title, keys in FIELD_GROUPS:
            if title in ("Image source", "Renderer and timing", "Raster and tone",
                         "Stochastic, stipple, fusion", "Output processing"):
                if run_mode != "app":
                    continue
            elif title in ("Live source", "Live sweep and capture",
                           "Live visualizer"):
                if run_mode != "live":
                    continue
            visible = []
            for key in keys:
                if key == "xy_dir" and self.settings.get("app_source") != "bake":
                    continue
                if key == "list_from_images" and self.settings.get("app_source") != "bake":
                    continue
                if key == "live_size" and self.settings.get("app_source") != "images":
                    continue
                if key == "video_file" and self.settings.get("live_source") != "video":
                    continue
                if key == "ffmpeg_input" and self.settings.get("live_source") not in ("ffmpeg", "camera"):
                    continue
                if key == "display" and self.settings.get("live_source") not in ("ffmpeg",):
                    continue
                if key == "region" and self.settings.get("live_source") not in (
                        "screen", "ffmpeg", "camera"):
                    continue
                if key in {"walk_radius", "walk_stride", "walk_reseed_ms",
                           "stochastic_gamma", "walk_edge", "walk_hz"} and \
                        self.settings.get("render_mode") not in ("stochastic", "fusion"):
                    continue
                if key == "fusion" and self.settings.get("render_mode") != "fusion":
                    continue
                if key == "stipple_points" and self.settings.get("render_mode") != "stipple":
                    continue
                if key in {"mix", "mix_duty"} and (
                        self.settings.get("app_source") != "bake" or
                        self.settings.get("render_mode") == "fusion"):
                    continue
                visible.append(key)
            if visible:
                groups.append((title, visible))
        return groups

    def _visible_fields(self):
        return [key for _title, keys in self._sections() for key in keys]

    def _persist(self):
        try:
            save_preferences(self.preference_path, self.settings, self.resume)
        except OSError as exc:
            self.notice = f"Could not save scope preferences: {exc}"

    def _resume_key(self):
        path = str(self.settings.get("video_file", "")).strip()
        if not path:
            return ""
        parsed = urlparse(path)
        if parsed.scheme and parsed.netloc:
            return ""  # live streams have no stable file resume position
        candidate = Path(path).expanduser()
        try:
            return str(candidate.resolve()) if candidate.exists() else path
        except OSError:
            return path

    def _remember_playback(self, persist=False):
        if not self.playback or not self._resume_key():
            return
        entry = dict(self.playback)
        entry["position"] = max(0.0, float(entry.get("position") or 0.0))
        self.resume[self._resume_key()] = entry
        now = time.monotonic()
        if persist or now - self._last_resume_save > 5:
            self._last_resume_save = now
            self._persist()

    def _assign(self, key, value):
        self.settings[key] = value
        if key in ("run_mode", "app_source", "live_source"):
            if key == "run_mode":
                self.selected = "app_source" if value == "app" else "live_source"
            elif key == "app_source" and value == "images" and self.settings.get("render_mode") not in dict(IMAGE_RENDER_CHOICES).values():
                self.settings["render_mode"] = "raster"
            elif key == "live_source":
                if value == "video" and not self.settings.get("video_file"):
                    self.selected = "video_file"
                elif value == "camera" and not self.settings.get("ffmpeg_input"):
                    self.selected = "ffmpeg_input"
        self.dropdown = None
        self._ensure_selection()
        if hasattr(self, "_height"):
            self._scroll_selection()
        self._persist()
        self.dirty = True

    def _ensure_selection(self):
        fields = self._visible_fields()
        if self.selected not in fields and fields:
            self.selected = fields[0]
            self.scroll = 0

    def _edit_start(self, key):
        self.selected = key
        self.editing = True
        self.edit_buffer = str(self.settings.get(key, ""))
        self.dirty = True

    def _edit_finish(self, commit=True):
        if not self.editing:
            return
        key, value = self.selected, self.edit_buffer.strip()
        self.editing = False
        if commit:
            if key in NUMBER_FIELDS and value:
                try:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError
                    integer_field = key in {
                        "live_size", "fps", "samples", "geometry_samples", "fields", "walk_radius",
                        "walk_stride", "stipple_points", "rows", "oversample",
                        "rotation", "live_fps", "live_samples", "live_fields",
                        "live_rows", "live_oversample", "downto",
                        "buffer_blocks", "blocksize"}
                    if integer_field and not number.is_integer():
                        raise ValueError
                    value = str(int(number) if integer_field else number)
                except ValueError:
                    self.notice = f"Invalid value for {FIELD_LABELS.get(key, key)}"
                    self.dirty = True
                    return
            if key == "channels":
                try:
                    value = ",".join(map(str, parse_channel_pair(value)))
                except ValueError as exc:
                    self.notice = str(exc)
                    self.dirty = True
                    return
            if key == "video_file" and str(self.settings.get(key, "")) != value:
                self.playback = None
                self.seek_preview = None
            self.settings[key] = value
            if key == "fields":
                self.settings["fields_explicit"] = bool(value)
            self._persist()
        self.dirty = True

    def _toggle(self, key):
        self.settings[key] = not bool(self.settings.get(key))
        self._persist()
        self.dirty = True

    def _current_resume(self):
        if not self._resume_key():
            return 0.0
        return self._playback_state()[0]

    def _build_command(self):
        outputs = self._device_choices()[1]
        return build_command(self.settings, outputs, root=ROOT,
                             video_start=self._current_resume())

    def _start(self):
        if self.editing:
            self._edit_finish()
        if self.process is not None:
            return
        try:
            command = self._build_command()
        except (ValueError, TypeError, OSError) as exc:
            self.notice = str(exc)
            self.dirty = True
            return
        kwargs = {
            "cwd": str(ROOT), "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
            "text": True, "encoding": "utf-8", "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(command, **kwargs)
        except (OSError, ValueError) as exc:
            self.notice = f"Could not start scope: {exc}"
            self.dirty = True
            return
        self.process = process
        self.active_video_transport = (
            self.settings.get("run_mode") == "live" and
            self.settings.get("live_source") == "video")
        self.playback = None
        self.seek_preview = None
        self.dropdown = None
        self.stop_requested = False
        self.lines = []
        self.page = "live"
        self.notice = "Starting scope…"
        self.reader = threading.Thread(target=self._read_process,
                                       args=(process,), daemon=True,
                                       name="scope-gui-output")
        self.reader.start()
        self._persist()
        self.dirty = True

    def _read_process(self, process):
        try:
            for line in process.stdout:
                self.events.put(("line", line.rstrip("\r\n")))
        except (OSError, ValueError) as exc:
            self.events.put(("line", f"Output reader: {exc}"))
        finally:
            self.events.put(("exit", process.wait()))

    def _is_control_process(self):
        return self.active_video_transport and self.process is not None

    def _has_video_selection(self):
        return (self.settings.get("run_mode") == "live" and
                self.settings.get("live_source") == "video" and
                bool(str(self.settings.get("video_file", "")).strip()))

    def _stop(self):
        process = self.process
        if process is None or process.poll() is not None or self.stop_requested:
            return
        self.stop_requested = True
        self.notice = "Stopping scope and closing audio…"
        self.dirty = True
        control = getattr(process, "stdin", None)
        graceful_control = self._is_control_process() and control is not None
        if graceful_control:
            try:
                control.write(json.dumps({"transport": "shutdown"}) + "\n")
                control.flush()
                control.close()
            except (OSError, ValueError):
                graceful_control = False
        if not graceful_control:
            try:
                if control is not None:
                    control.close()
            except (OSError, ValueError):
                pass
            try:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.send_signal(signal.SIGINT)
            except (OSError, ValueError):
                try:
                    process.terminate()
                except OSError:
                    pass
        threading.Thread(target=self._stop_timeout, args=(process,),
                         daemon=True, name="scope-gui-stop-timeout").start()

    def _stop_timeout(self, process):
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            process.terminate()
            process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass

    def _transport(self, command, position=None):
        process = self.process
        if process is None:
            if not self._has_video_selection():
                return
            if command not in ("seek", "restart"):
                return
            current_position, duration, _paused = self._playback_state()
            if command == "seek":
                try:
                    current_position = max(0.0, float(position))
                except (TypeError, ValueError):
                    return
                if duration:
                    current_position = min(current_position, duration)
            else:
                current_position = 0.0
            self.playback = {"position": current_position, "duration": duration,
                             "paused": False}
            self._remember_playback(persist=True)
            self.dirty = True
            return
        if not self._is_control_process():
            return
        control = getattr(process, "stdin", None)
        if control is None:
            return
        message = {"transport": command}
        if position is not None:
            message["position"] = float(position)
        try:
            control.write(json.dumps(message) + "\n")
            control.flush()
        except (OSError, ValueError) as exc:
            self.notice = f"Video control failed: {exc}"
        if command in ("seek", "restart", "pause"):
            if self.playback is None:
                self.playback = {"position": 0.0, "duration": None,
                                 "paused": command == "pause"}
            if command == "seek":
                self.playback["position"] = float(position)
            elif command == "restart":
                self.playback["position"] = 0.0
            elif command == "pause":
                self.playback["paused"] = True
            self._remember_playback(persist=True)
        elif command == "play" and self.playback is not None:
            self.playback["paused"] = False
        self.dirty = True

    def _drain_events(self):
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "line":
                try:
                    record = json.loads(value)
                except (TypeError, ValueError):
                    record = None
                if isinstance(record, dict) and record.get("status") == "playback":
                    self.playback = record
                    self._remember_playback()
                elif isinstance(record, dict) and record.get("status") == "control":
                    if not record.get("ok"):
                        self.notice = str(record.get("message", "Video control failed"))
                else:
                    self.lines.append(value)
                    self.lines = self.lines[-26:]
                    if value:
                        self.notice = value
            elif kind == "exit":
                if self.process is not None:
                    self.process = None
                    self.active_video_transport = False
                    self.stop_requested = False
                    self._remember_playback(persist=True)
                    if self.close_when_stopped:
                        self.closed = True
                    else:
                        self.notice = ("Scope stopped." if value == 0 else
                                       f"Scope exited with status {value}.")
            self.dirty = True

    def _items(self):
        items = []
        for title, keys in self._sections():
            items.append(("header", title))
            items.extend(("field", key) for key in keys)
        return items

    def _visible_items(self):
        items = self._items()
        limit = self._visible_item_limit()
        if self.selected not in self._visible_fields():
            self._ensure_selection()
        self.scroll = max(0, min(self.scroll, max(0, len(items) - limit)))
        return items[self.scroll:self.scroll + limit], self.scroll

    def _visible_item_limit(self):
        return max(1, (self._height - self.TOOLBAR_HEIGHT
                       - self.FOOTER_HEIGHT - self._u(25))
                   // self.ROW_HEIGHT)

    def _draw(self):
        width, height = self.glfw.get_window_size(self.window)
        if width <= 0 or height <= 0:
            return
        self._height = height
        self._apply_ui_scale(height)
        unit = self._u
        image = self.Image.new("RGBA", (width, height), (12, 19, 26, 255))
        draw = self.ImageDraw.Draw(image)
        self.hits = {}
        toolbar_height = self.TOOLBAR_HEIGHT
        draw.rectangle((0, 0, width, toolbar_height), fill=(10, 18, 25, 255))
        toolbar = (("setup", "Setup", "Setup"),
                   ("live", "Live", "Live"),
                   ("start", "Stop" if self.process else "Start",
                    "Stop" if self.process else "Start"),
                   ("defaults", "Defaults", "Default"),
                   ("refresh", "Refresh devices", "Devices"),
                   ("fullscreen", "Restore" if self.fullscreen else "Fullscreen",
                    "Restore" if self.fullscreen else "Full"))
        compact = width < 980
        right = width - unit(12)
        compact_gutter = unit(5)
        full_gutter = unit(8)
        for key, full_label, short_label in reversed(toolbar):
            label = short_label if compact else full_label
            text_width = int(self.small.getlength(label))
            button_width = max(unit(48), text_width + unit(24))
            x = right - button_width
            rect = (x, unit(9), right, toolbar_height - unit(9))
            active = ((key == self.page) or (key == "start" and self.process))
            color = ((42, 85, 108) if key == "start" and self.process else
                     (43, 94, 123) if key == "start" else
                     (43, 94, 123) if key == "fullscreen" and self.fullscreen else
                     (39, 67, 86) if active else (22, 35, 46))
            draw.rounded_rectangle(rect, radius=unit(5), fill=color,
                                   outline=(67, 100, 122))
            draw.text((x + (button_width - text_width) / 2,
                       unit(20)), label, font=self.small,
                      fill=(246, 240, 235) if key == "start" else (236, 242, 247))
            self.hits[key] = rect
            right = x - (compact_gutter if compact else full_gutter)
        title = "SCOPE  ·  SETUP / LIVE"
        title_width = max(unit(54), right - unit(24))
        shown_title = self._fit(title, self.font, title_width)
        draw.text((unit(18), unit(18)), shown_title, font=self.font,
                  fill=(239, 245, 249))
        if self.page == "setup":
            self._draw_setup(draw, width, height)
        else:
            self._draw_live(draw, width, height)
        footer_top = height - self.FOOTER_HEIGHT
        draw.rectangle((0, footer_top, width, height), fill=(9, 15, 20, 255))
        help_text = (self.notice or
                     "Scroll or ↑/↓ settings · Enter edit · ←/→ choose · "
                     "Space start/stop · F11 fullscreen · Esc stop/close")
        draw.text((unit(14), footer_top + unit(13)),
                  self._fit(help_text, self.small, width - unit(28)),
                  font=self.small, fill=(151, 169, 179))
        framebuffer = self.glfw.get_framebuffer_size(self.window)
        if framebuffer[0] <= 0 or framebuffer[1] <= 0:
            return
        self.context.viewport = (0, 0, framebuffer[0], framebuffer[1])
        self.context.clear(0.04, 0.06, 0.08, 1.0)
        raw = image.tobytes()
        if self.texture is None or self.texture.size != image.size:
            if self.texture is not None:
                self.texture.release()
            self.texture = self.context.texture(image.size, 4, raw, alignment=1)
            self.texture.filter = (self.moderngl.LINEAR, self.moderngl.LINEAR)
        else:
            self.texture.write(raw, alignment=1)
        self.texture.use(location=0)
        self.vao.render(mode=self.moderngl.TRIANGLES, vertices=3)
        self.glfw.swap_buffers(self.window)

    def _draw_setup(self, draw, width, height):
        items, offset = self._visible_items()
        unit = self._u
        y = self.TOOLBAR_HEIGHT + unit(10)
        left, right = unit(18), width - unit(34)
        value_left = left + min(unit(328), round((right - left) * 0.38))
        for index, (kind, value) in enumerate(items):
            top, bottom = y, y + self.ROW_HEIGHT - 2
            if kind == "header":
                draw.text((left + unit(4), top + unit(5)), value, font=self.small,
                          fill=(134, 169, 188))
                draw.line((left + unit(160), top + unit(15), right,
                           top + unit(15)), fill=(47, 68, 83), width=unit(1))
            else:
                selected = value == self.selected
                rect = (left, top, right, bottom)
                draw.rounded_rectangle(
                    rect, radius=unit(4),
                    fill=(35, 60, 77) if selected else (17, 29, 39),
                    outline=(74, 111, 134) if selected else (32, 48, 60))
                label = ("Camera device" if value == "ffmpeg_input" and
                         self.settings.get("live_source") == "camera" else
                         FIELD_LABELS.get(value, value))
                draw.text((left + unit(10), top + unit(7)),
                          self._fit(label, self.small,
                                    value_left - left - unit(20)),
                          font=self.small, fill=(205, 218, 228))
                field_right = right - unit(12)
                if value in FILE_FIELDS or value in DIR_FIELDS:
                    browse = (right - unit(82), top + unit(3),
                              right - unit(5), bottom - unit(3))
                    draw.rounded_rectangle(browse, radius=unit(4),
                                           fill=(30, 58, 76),
                                           outline=(75, 111, 132))
                    draw.text((browse[0] + unit(9), top + unit(7)),
                              "Browse", font=self.small,
                              fill=(229, 239, 246))
                    self.hits[f"browse:{value}"] = browse
                    field_right = browse[0] - unit(8)
                display = (self.edit_buffer if self.editing and value == self.selected
                           else "" if value in BOOL_FIELDS
                           else self._display_value(value))
                draw.text((value_left, top + unit(7)),
                          self._fit(display, self.small,
                                    field_right - value_left - unit(18)),
                          font=self.small, fill=(243, 208, 146) if selected else
                          (237, 242, 246))
                if self._is_dropdown_field(value):
                    draw.text((field_right - unit(12), top + unit(6)),
                              "▾", font=self.small,
                              fill=(134, 169, 188))
                elif value in BOOL_FIELDS:
                    state = "On" if self.settings.get(value) else "Off"
                    draw.text((field_right - unit(35), top + unit(7)),
                              state, font=self.small,
                              fill=(160, 212, 180) if self.settings.get(value)
                              else (133, 159, 177))
                self.hits[f"field:{value}"] = rect
            y += self.ROW_HEIGHT
        all_items = self._items()
        limit = self._visible_item_limit()
        if len(all_items) > limit:
            track_top = self.TOOLBAR_HEIGHT + unit(10)
            track_bottom = height - self.FOOTER_HEIGHT - unit(8)
            track_x = width - unit(14)
            track_height = max(1, track_bottom - track_top)
            draw.rounded_rectangle(
                (track_x - unit(4), track_top, track_x + unit(4),
                 track_bottom), radius=unit(4), fill=(28, 43, 54))
            thumb_height = max(
                unit(24), round(track_height * min(1.0,
                                                    limit / len(all_items))))
            max_scroll = max(1, len(all_items) - limit)
            thumb_top = track_top + round(
                (track_height - thumb_height) * offset / max_scroll)
            draw.rounded_rectangle(
                (track_x - unit(4), thumb_top, track_x + unit(4),
                 thumb_top + thumb_height), radius=unit(4),
                fill=(70, 106, 127))
        if self.dropdown:
            self._draw_dropdown(draw, width, height)

    def _draw_dropdown(self, draw, width, height):
        unit = self._u
        key = self.dropdown
        choices = self._choices(key)
        rect = self.hits.get(f"field:{key}")
        if not rect:
            return
        menu_width = min(max(unit(300), rect[2] - rect[0]), width - unit(36))
        left = max(unit(18), width - unit(18) - menu_width)
        item_height = unit(29)
        visible = min(9, len(choices), max(
            1, (height - self.TOOLBAR_HEIGHT - self.FOOTER_HEIGHT)
            // item_height))
        top = rect[3] + unit(2)
        if top + visible * item_height > height - self.FOOTER_HEIGHT:
            top = max(self.TOOLBAR_HEIGHT, rect[1] - visible * item_height)
        current = str(self.settings.get(key, ""))
        for index, (label, value) in enumerate(choices[:visible]):
            item = (left, top + index * item_height,
                    left + menu_width, top + (index + 1) * item_height)
            selected = str(value) == current
            draw.rectangle(item, fill=(39, 67, 86) if selected else (18, 30, 39),
                           outline=(54, 77, 92))
            draw.text((left + unit(10), item[1] + unit(6)),
                      self._fit(label, self.small, menu_width - unit(20)),
                      font=self.small, fill=(235, 242, 247))
            self.hits[f"option:{index}"] = item

    def _draw_live(self, draw, width, height):
        unit = self._u
        top = self.TOOLBAR_HEIGHT + unit(14)
        is_video = self._has_video_selection()
        if is_video:
            position, duration, paused = self._playback_state()
            active = self._is_control_process()
            controls = (("play_pause", ("Play" if paused else "Pause")
                         if active else ("Resume" if position > 0 else "Start")),
                        ("restart", "Restart"))
            x = 20
            for key, label in controls:
                rect = (x, top, x + unit(100), top + unit(34))
                draw.rounded_rectangle(rect, radius=unit(4), fill=(24, 48, 63),
                                       outline=(65, 103, 124))
                draw.text((x + unit(14), top + unit(9)), label, font=self.small,
                          fill=(232, 241, 246))
                self.hits[key] = rect
                x += unit(112)
            if duration:
                bar = (unit(250), top + unit(10), width - unit(125),
                       top + unit(25))
                draw.rounded_rectangle(bar, radius=unit(5), fill=(49, 67, 78))
                fraction = min(1.0, max(0.0, position / duration))
                marker_x = round(bar[0] + fraction * (bar[2] - bar[0]))
                draw.rounded_rectangle((bar[0], bar[1], marker_x, bar[3]),
                                       radius=unit(5), fill=(63, 137, 174))
                draw.ellipse((marker_x - unit(6), bar[1] - unit(4),
                              marker_x + unit(6), bar[3] + unit(4)),
                             fill=(205, 232, 245))
                self.hits["seek"] = (bar[0], top, bar[2], top + unit(32))
                text = f"{self._clock(position)} / {self._clock(duration)}"
                draw.text((width - unit(116), top + unit(9)), text, font=self.small,
                          fill=(203, 219, 228))
            top += unit(52)
        draw.text((unit(20), top), "Scope process output", font=self.font,
                  fill=(231, 240, 246))
        top += unit(32)
        draw.rounded_rectangle((unit(18), top, width - unit(18),
                                height - self.FOOTER_HEIGHT - unit(38)),
                               radius=unit(5), fill=(6, 10, 13),
                               outline=(49, 69, 83))
        max_lines = max(1, (height - top - self.FOOTER_HEIGHT - unit(58))
                        // unit(19))
        for index, line in enumerate(self.lines[-max_lines:]):
            draw.text((unit(30), top + unit(11) + index * unit(19)),
                      self._fit(line, self.small, width - unit(60)),
                      font=self.small, fill=(183, 201, 212))

    def _display_value(self, key):
        value = self.settings.get(key, "")
        if key == "device":
            choices = self._choices(key)
            return next((label for label, choice in choices
                         if str(choice) == str(value)), "Choose an output")
        if key == "ffmpeg_input" and self.settings.get("live_source") == "camera":
            if not value:
                return "Choose a camera"
            choices = self._camera_cache[0] if self._camera_cache else ()
            return next((label for label, choice in choices
                         if str(choice) == str(value)), str(value))
        if key in SELECT_CHOICES:
            return next((label for label, choice in self._choices(key)
                         if str(choice) == str(value)), str(value))
        if key in BOOL_FIELDS:
            return "On" if value else "Off"
        return str(value) if value is not None else ""

    @staticmethod
    def _fit(text, font, width):
        text = str(text)
        while text and text != "…" and font.getlength(text) > max(1, width):
            text = text[:-2] + "…"
        return text

    @staticmethod
    def _clock(seconds):
        seconds = max(0, int(float(seconds or 0)))
        return f"{seconds // 60}:{seconds % 60:02d}"

    def _playback_state(self):
        if self.playback is None:
            saved = self.resume.get(self._resume_key(), {})
        else:
            saved = self.playback
        try:
            position = float(saved.get("position", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            position = 0.0
        if not math.isfinite(position) or position < 0:
            position = 0.0
        try:
            duration = float(saved.get("duration"))
        except (AttributeError, TypeError, ValueError):
            duration = None
        if duration is not None and (not math.isfinite(duration) or duration <= 0):
            duration = None
        if self.seek_preview is not None:
            position = self.seek_preview
        return position, duration, bool(saved.get("paused", False))

    def _seek_at(self, x):
        rect = self.hits.get("seek")
        if not rect:
            return
        _position, duration, _paused = self._playback_state()
        if duration:
            fraction = min(1.0, max(0.0, (x - rect[0]) / max(1, rect[2] - rect[0])))
            self._transport("seek", fraction * duration)

    def _on_cursor(self, _window, x, _y):
        if self.seek_drag is None:
            return
        rect = self.hits.get("seek")
        if not rect:
            return
        _position, duration, _paused = self._playback_state()
        if duration:
            fraction = min(1.0, max(0.0, (x - rect[0]) / max(1, rect[2] - rect[0])))
            self.seek_preview = fraction * duration
            self.dirty = True

    def _on_mouse(self, window, button, action, _mods):
        if button != self.glfw.MOUSE_BUTTON_LEFT:
            return
        x, y = self.glfw.get_cursor_pos(window)
        if action == self.glfw.RELEASE and self.seek_drag is not None:
            self.seek_drag = None
            self._seek_at(x)
            self.seek_preview = None
            return
        if action != self.glfw.PRESS:
            return
        if self.editing:
            self._edit_finish()
        hit_items = list(self.hits.items())
        if self.dropdown:
            hit_items.sort(key=lambda item: 0 if item[0].startswith("option:") else 1)
        for key, rect in hit_items:
            if not (rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]):
                continue
            if key.startswith("option:") and self.dropdown:
                options = self._choices(self.dropdown)
                index = int(key.split(":", 1)[1])
                if index < len(options):
                    self._assign(self.dropdown, options[index][1])
            elif key.startswith("browse:"):
                if self.process is not None:
                    self.notice = "Stop scope before changing source paths or settings."
                    self.dirty = True
                    return
                self._browse(key.split(":", 1)[1])
            elif key.startswith("field:"):
                if self.process is not None:
                    self.notice = "Stop scope before changing its settings."
                    self.dirty = True
                    return
                field = key.split(":", 1)[1]
                self.selected = field
                if self.dropdown:
                    self.dropdown = None
                if field in BOOL_FIELDS:
                    self._toggle(field)
                elif self._is_dropdown_field(field):
                    self._open_dropdown(field)
                else:
                    self._edit_start(field)
            elif key == "setup":
                self.page = "setup"
            elif key == "live":
                self.page = "live"
            elif key == "start":
                self._stop() if self.process else self._start()
            elif key == "fullscreen":
                self._toggle_fullscreen()
            elif key == "defaults":
                if self.process is not None:
                    self.notice = "Stop scope before resetting settings."
                    self.dirty = True
                    return
                self.settings = dict(DEFAULT_SETTINGS)
                self._persist()
                self.notice = "Settings reset to application defaults."
            elif key == "refresh":
                self._device_choices(refresh=True)
                cameras, camera_error = self._camera_choices(refresh=True)
                if (self.settings.get("run_mode") == "live" and
                        self.settings.get("live_source") == "camera"):
                    self.notice = (self._camera_notice() if camera_error else
                                   f"Refreshed {len(cameras)} camera device(s).")
                else:
                    self.notice = "Audio and video device lists refreshed."
            elif key == "play_pause":
                if self.process is None:
                    self._start()
                else:
                    _position, _duration, paused = self._playback_state()
                    self._transport("play" if paused else "pause")
            elif key == "restart":
                self._transport("restart")
            elif key == "seek":
                self.seek_drag = True
                self.seek_preview = None
            self.dirty = True
            return

    def _browse(self, key):
        if self.process is not None:
            self.notice = "Stop scope before changing source paths."
            return
        try:
            path = _file_picker(self.settings.get(key, ""), directory=key in DIR_FIELDS)
        except (OSError, RuntimeError) as exc:
            self.notice = str(exc)
            return
        if path:
            if key == "video_file" and str(self.settings.get(key, "")) != path:
                self.playback = None
                self.seek_preview = None
            self.settings[key] = path
            self.selected = key
            if key == "video_file":
                self.settings["run_mode"] = "live"
                self.settings["live_source"] = "video"
            if hasattr(self, "_height"):
                self._scroll_selection()
            self._persist()
            self.notice = f"Selected {Path(path).name}"
        self.dirty = True

    def _on_scroll(self, _window, _x, y):
        if self.page == "setup":
            items = self._items()
            limit = self._visible_item_limit()
            amount = max(1, round(abs(y) * 3)) if y else 0
            self.scroll = max(0, min(
                max(0, len(items) - limit),
                self.scroll - (amount if y > 0 else -amount)))
            self.dirty = True

    def _on_drop(self, _window, paths):
        if not paths or self.process is not None:
            return
        path = Path(paths[0]).expanduser()
        if path.is_dir():
            self.settings["run_mode"] = "app"
            self.settings["app_source"] = "images"
            self.settings["image_dir"] = str(path)
            if self.settings.get("render_mode") not in dict(IMAGE_RENDER_CHOICES).values():
                self.settings["render_mode"] = "raster"
            self.selected = "image_dir"
            self.notice = f"Selected image folder: {path.name}"
        else:
            if str(self.settings.get("video_file", "")) != str(path):
                self.playback = None
                self.seek_preview = None
            self.settings["run_mode"] = "live"
            self.settings["live_source"] = "video"
            self.settings["video_file"] = str(path)
            self.selected = "video_file"
            self.notice = f"Selected video: {path.name}"
        if hasattr(self, "_height"):
            self._scroll_selection()
        self._persist()
        self.dirty = True

    def _on_char(self, _window, codepoint):
        if self.editing and 32 <= codepoint <= 0x10FFFF:
            try:
                char = chr(codepoint)
            except ValueError:
                return
            if char.isprintable():
                self.edit_buffer += char
                self.dirty = True

    def _on_key(self, _window, key, _scancode, action, mods):
        if action not in (self.glfw.PRESS, self.glfw.REPEAT):
            return
        if self.editing:
            if key in (self.glfw.KEY_ENTER, self.glfw.KEY_KP_ENTER):
                self._edit_finish()
            elif key == self.glfw.KEY_ESCAPE:
                self._edit_finish(commit=False)
            elif key == self.glfw.KEY_BACKSPACE:
                self.edit_buffer = self.edit_buffer[:-1]
            elif key == self.glfw.KEY_A and mods & (self.glfw.MOD_CONTROL |
                                                    getattr(self.glfw, "MOD_SUPER", 0)):
                self.edit_buffer = ""
            elif key == self.glfw.KEY_V and mods & (self.glfw.MOD_CONTROL |
                                                    getattr(self.glfw, "MOD_SUPER", 0)):
                try:
                    self.edit_buffer += self.glfw.get_clipboard_string(self.window) or ""
                except Exception:
                    pass
            self.dirty = True
            return
        if self.process is not None and key not in (
                self.glfw.KEY_ESCAPE, self.glfw.KEY_SPACE, self.glfw.KEY_TAB,
                self.glfw.KEY_F11):
            return
        if key == self.glfw.KEY_ESCAPE:
            if self.dropdown:
                self.dropdown = None
            elif self.process:
                self._stop()
            else:
                self.closed = True
        elif key == self.glfw.KEY_SPACE:
            self._stop() if self.process else self._start()
        elif key == self.glfw.KEY_TAB:
            self.page = "live" if self.page == "setup" else "setup"
        elif key == self.glfw.KEY_UP or key == self.glfw.KEY_DOWN:
            if self.dropdown:
                options = self._choices(self.dropdown)
                values = [str(value) for _label, value in options]
                current = str(self.settings.get(self.dropdown, ""))
                try:
                    index = values.index(current)
                except ValueError:
                    index = 0
                index = (index + (-1 if key == self.glfw.KEY_UP else 1)) % max(1, len(options))
                if options:
                    self._assign(self.dropdown, options[index][1])
                    self.dropdown = None
            else:
                fields = self._visible_fields()
                try:
                    index = fields.index(self.selected)
                except ValueError:
                    index = 0
                index = max(0, min(len(fields) - 1,
                                   index + (-1 if key == self.glfw.KEY_UP else 1)))
                if fields:
                    self.selected = fields[index]
                    self._scroll_selection()
        elif key in (self.glfw.KEY_LEFT, self.glfw.KEY_RIGHT):
            if self.selected in BOOL_FIELDS:
                self._toggle(self.selected)
            elif self._is_dropdown_field(self.selected):
                options = self._choices(self.selected)
                values = [str(value) for _label, value in options]
                current = str(self.settings.get(self.selected, ""))
                try:
                    index = values.index(current)
                except ValueError:
                    index = 0
                index = (index + (1 if key == self.glfw.KEY_RIGHT else -1)) % max(1, len(options))
                if options:
                    self._assign(self.selected, options[index][1])
                else:
                    self._open_dropdown(self.selected)
        elif key in (self.glfw.KEY_ENTER, self.glfw.KEY_KP_ENTER):
            if self.selected in BOOL_FIELDS:
                self._toggle(self.selected)
            elif self._is_dropdown_field(self.selected):
                self._open_dropdown(self.selected)
            else:
                self._edit_start(self.selected)
        elif key == self.glfw.KEY_F11:
            self._toggle_fullscreen()
        self.dirty = True

    def _scroll_selection(self):
        items = self._items()
        selected_index = next((index for index, item in enumerate(items)
                               if item == ("field", self.selected)), 0)
        limit = self._visible_item_limit()
        if selected_index < self.scroll:
            self.scroll = selected_index
        elif selected_index >= self.scroll + limit:
            self.scroll = selected_index - limit + 1

    def _toggle_fullscreen(self):
        if self.fullscreen:
            if self._windowed_bounds is None:
                area = self.work_area
                area_size = area[2:] if area is not None else self.WINDOW_SIZE
                width, height = window_size_for_workarea(
                    *area_size, desired=self.WINDOW_SIZE)
                if area is None:
                    x = y = 40
                else:
                    x = area[0] + max(0, (area[2] - width) // 2)
                    y = area[1] + max(0, (area[3] - height) // 2)
            else:
                x, y, width, height = self._windowed_bounds
            self.glfw.set_window_monitor(
                self.window, None, x, y, width, height, self.glfw.DONT_CARE)
            self.fullscreen = False
        else:
            monitor = self.primary_monitor or self.glfw.get_primary_monitor()
            if monitor is not None:
                x, y = self.glfw.get_window_pos(self.window)
                width, height = self.glfw.get_window_size(self.window)
                self._windowed_bounds = (int(x), int(y),
                                         int(width), int(height))
                mode = self.glfw.get_video_mode(monitor)
                self.glfw.set_window_monitor(self.window, monitor, 0, 0,
                                             mode.size.width, mode.size.height,
                                             mode.refresh_rate)
                self.fullscreen = True
        focus_window = getattr(self.glfw, "focus_window", None)
        if focus_window is not None:
            focus_window(self.window)
        self.dirty = True

    def run(self):
        self.glfw.make_context_current(self.window)
        while not self.glfw.window_should_close(self.window) and not self.closed:
            self.glfw.poll_events()
            window_size = tuple(int(value) for value in
                                self.glfw.get_window_size(self.window))
            framebuffer_size = tuple(int(value) for value in
                                     self.glfw.get_framebuffer_size(self.window))
            sizes = window_size, framebuffer_size
            if sizes != self._last_sizes:
                self._last_sizes = sizes
                if window_size[1] > 0:
                    self._height = window_size[1]
                self.dirty = True
            self._drain_events()
            if self.process is not None and self.process.poll() is not None:
                self.events.put(("exit", self.process.returncode))
                self._drain_events()
            if self.dirty:
                self._draw()
                self.dirty = False
            self.glfw.wait_events_timeout(0.05)
        if self.process is not None:
            self.close_when_stopped = True
            self._stop()
            deadline = time.monotonic() + 6.0
            while self.process is not None and time.monotonic() < deadline:
                self.glfw.poll_events()
                self._drain_events()
                time.sleep(0.03)
            if self.process is not None:
                try:
                    self.process.kill()
                except OSError:
                    pass
        self.close()

    def close(self):
        if self.closed and self.window is None:
            return
        self.closed = True
        self._persist()
        for resource in (self.texture, self.vao, self.program, self.context):
            try:
                if resource is not None:
                    resource.release()
            except Exception:
                pass
        try:
            self.glfw.destroy_window(self.window)
            self.glfw.terminate()
        except Exception:
            pass
        self.window = None


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Scope setup and live controller")
    parser.add_argument("--no-restore", action="store_true",
                        help="ignore saved scope GUI preferences")
    args = parser.parse_args(argv)
    gui = ScopeLauncher(restore_preferences=not args.no_restore)
    gui.run()


if __name__ == "__main__":
    main()
