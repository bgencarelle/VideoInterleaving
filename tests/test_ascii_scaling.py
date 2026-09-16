#!/usr/bin/env python3
"""
Helper to visualize the COVER/CROP math used by ascii_converter.to_ascii.
No padding is applied; the scaled image always fills the terminal grid and any
overflow is cropped symmetrically.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Utility script; run directly instead of under pytest")


def test_ascii_scaling(image_w, image_h, terminal_cols, terminal_rows, font_ratio=0.5, test_name=""):
    print("=" * 80)
    if test_name:
        print(f"ASCII Scaling Test: {test_name}")
    else:
        print("ASCII Scaling Test")
    print("=" * 80)
    print(f"Image dimensions: {image_w}x{image_h}")
    print(f"Terminal dimensions: {terminal_cols}x{terminal_rows}")
    print(f"Font ratio: {font_ratio}")
    print()

    # Calculate aspect ratios
    image_aspect = image_w / image_h if image_h else 1.0
    terminal_display_aspect = (terminal_cols * font_ratio) / terminal_rows if terminal_rows else 1.0

    print("Aspect Ratios:")
    print(f"  Image aspect ratio: {image_aspect:.4f} ({image_w}/{image_h})")
    print(f"  Terminal display aspect ratio: {terminal_display_aspect:.4f} (({terminal_cols} * {font_ratio}) / {terminal_rows})")
    print()

    # COVER scale: pick the larger scale so both dimensions meet or exceed the target
    scale_x = terminal_cols / image_w if image_w else 1.0
    scale_y = terminal_rows / (image_h * font_ratio) if image_h else 1.0
    scale = max(scale_x, scale_y)

    scaled_w = max(1, int(image_w * scale))
    scaled_h = max(1, int(image_h * scale * font_ratio))

    # Crop overflow evenly
    crop_x = max(0, (scaled_w - terminal_cols) // 2)
    crop_y = max(0, (scaled_h - terminal_rows) // 2)

    print("Scale and Crop:")
    print(f"  scale_x = terminal_cols / image_w = {terminal_cols} / {image_w} = {scale_x:.6f}")
    print(f"  scale_y = terminal_rows / (image_h * font_ratio) = {terminal_rows} / ({image_h} * {font_ratio}) = {scale_y:.6f}")
    print(f"  Using COVER scale = max(scale_x, scale_y) = {scale:.6f}")
    print()
    print(f"  Scaled dimensions (pre-crop): {scaled_w} x {scaled_h}")
    print(f"  Crop offsets: {crop_x}px left/right, {crop_y}px top/bottom")
    print(f"  Final terminal grid: {terminal_cols} x {terminal_rows}")

    print("=" * 80)

    return {
        'scale': scale,
        'scaled_w': scaled_w,
        'scaled_h': scaled_h,
        'crop_x': crop_x,
        'crop_y': crop_y,
        'image_aspect': image_aspect,
        'terminal_aspect': terminal_display_aspect
    }


if __name__ == '__main__':
    # Test 1: Standard case - 720x960 image in 90x60 terminal (scaling down)
    print("\n")
    test_ascii_scaling(720, 960, 90, 60, 0.5, "720x960 -> 90x60 (scale down)")

    # Test 2: Image smaller than terminal (scaling up)
    print("\n")
    test_ascii_scaling(45, 60, 90, 60, 0.5, "45x60 -> 90x60 (scale up)")

    # Test 3: Very small image (scaling up significantly)
    print("\n")
    test_ascii_scaling(30, 40, 90, 60, 0.5, "30x40 -> 90x60 (scale up significantly)")

    # Test 4: Wide image (different aspect ratio)
    print("\n")
    test_ascii_scaling(960, 720, 90, 60, 0.5, "960x720 -> 90x60 (wide image)")
