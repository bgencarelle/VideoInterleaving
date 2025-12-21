#!/usr/bin/env python3
"""
Test script to calculate correct scale factor and padding for ASCII conversion.
Tests with a 720x960 image in a 90x60 terminal, emulating the standard image_display code.
Also tests scaling up when image is smaller than terminal.
"""

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
    image_aspect = image_w / image_h
    terminal_raw_aspect = terminal_cols / terminal_rows
    terminal_display_aspect = (terminal_cols * font_ratio) / terminal_rows
    
    print("Aspect Ratios:")
    print(f"  Image aspect ratio: {image_aspect:.4f} ({image_w}/{image_h})")
    print(f"  Terminal raw aspect ratio: {terminal_raw_aspect:.4f} ({terminal_cols}/{terminal_rows})")
    print(f"  Terminal display aspect ratio: {terminal_display_aspect:.4f} (({terminal_cols} * {font_ratio}) / {terminal_rows})")
    print()
    
    # Standard image_display approach (from renderer.py composite_cpu):
    # scale = min(target_w / tw, target_h / th)
    # new_w = int(tw * scale)
    # new_h = int(th * scale)
    # Then center with padding
    
    # For ASCII, we need to account for font_ratio in the height constraint
    # The terminal has terminal_cols columns and terminal_rows rows
    # Each character cell is 1 pixel in the output array
    # But when displayed, the aspect ratio is affected by font_ratio
    
    # Calculate scale factors
    # We want the DISPLAYED aspect ratio to match the image aspect ratio
    # Display aspect = (scaled_w * font_ratio) / scaled_h = image_aspect
    # And: scaled_w <= terminal_cols, scaled_h <= terminal_rows
    # We want to FILL one dimension completely to optimize space usage
    
    # Option 1: Fill width completely
    # scaled_w = terminal_cols
    # (terminal_cols * font_ratio) / scaled_h = image_aspect
    # scaled_h = (terminal_cols * font_ratio) / image_aspect
    scaled_h_if_fill_width = (terminal_cols * font_ratio) / image_aspect
    scale_x = scaled_h_if_fill_width / image_h if image_h > 0 else 1.0
    
    # Option 2: Fill height completely
    # scaled_h = terminal_rows
    # (scaled_w * font_ratio) / terminal_rows = image_aspect
    # scaled_w = (terminal_rows * image_aspect) / font_ratio
    scaled_w_if_fill_height = (terminal_rows * image_aspect) / font_ratio
    scale_y = scaled_w_if_fill_height / image_w if image_w > 0 else 1.0
    
    # Use FIT: choose the scale that ensures both dimensions fit AND fills one dimension
    # We want to maximize space usage, so we fill the dimension that fits
    if scaled_h_if_fill_width <= terminal_rows:
        # Filling width fits - use this to maximize width usage
        # Set width to terminal_cols and calculate height to maintain display aspect ratio
        scaled_w_pixels = terminal_cols
        scaled_h_pixels = int((terminal_cols * font_ratio) / image_aspect) if image_aspect > 0 else terminal_rows
        # Calculate the actual scale factor used
        scale = scaled_w_pixels / image_w if image_w > 0 else 1.0
    else:
        # Must fit height instead - fill height completely
        # Set height to terminal_rows and calculate width to maintain display aspect ratio
        scaled_h_pixels = terminal_rows
        scaled_w_pixels = int((terminal_rows * image_aspect) / font_ratio) if font_ratio > 0 else terminal_cols
        # Calculate the actual scale factor used
        scale = scaled_h_pixels / image_h if image_h > 0 else 1.0
    
    # Clamp to terminal bounds (safety check)
    scaled_w_pixels = max(1, min(scaled_w_pixels, terminal_cols))
    scaled_h_pixels = max(1, min(scaled_h_pixels, terminal_rows))
    
    print("Scale Factor Calculation:")
    print(f"  Option 1 (fill width):")
    print(f"    scaled_h = (terminal_cols * font_ratio) / image_aspect = ({terminal_cols} * {font_ratio}) / {image_aspect:.4f} = {scaled_h_if_fill_width:.2f}")
    print(f"    scale_x = scaled_h / image_h = {scaled_h_if_fill_width:.2f} / {image_h} = {scale_x:.6f}")
    print(f"    Fits height? {scaled_h_if_fill_width:.2f} <= {terminal_rows}: {'✓' if scaled_h_if_fill_width <= terminal_rows else '✗'}")
    print()
    print(f"  Option 2 (fill height):")
    print(f"    scaled_w = (terminal_rows * image_aspect) / font_ratio = ({terminal_rows} * {image_aspect:.4f}) / {font_ratio} = {scaled_w_if_fill_height:.2f}")
    print(f"    scale_y = scaled_w / image_w = {scaled_w_if_fill_height:.2f} / {image_w} = {scale_y:.6f}")
    print()
    
    # Use FIT: choose the scale that fits
    if scaled_h_if_fill_width <= terminal_rows:
        print(f"  Using Option 1 (fill width): scale = {scale:.6f}")
    else:
        print(f"  Using Option 2 (fill height): scale = {scale:.6f}")
    print()
    
    print("Scaled Dimensions:")
    print(f"  scaled_w_pixels = int(image_w * scale) = int({image_w} * {scale:.6f}) = {scaled_w_pixels}")
    print(f"  scaled_h_pixels = int(image_h * scale) = int({image_h} * {scale:.6f}) = {scaled_h_pixels}")
    print()
    
    # Verify aspect ratio is preserved (in display space, not pixel space)
    scaled_pixel_aspect = scaled_w_pixels / scaled_h_pixels if scaled_h_pixels > 0 else 1.0
    scaled_display_aspect = (scaled_w_pixels * font_ratio) / scaled_h_pixels if scaled_h_pixels > 0 else 1.0
    print(f"  Scaled pixel aspect ratio: {scaled_pixel_aspect:.4f} ({scaled_w_pixels}/{scaled_h_pixels})")
    print(f"  Scaled display aspect ratio: {scaled_display_aspect:.4f} (({scaled_w_pixels} * {font_ratio}) / {scaled_h_pixels})")
    print(f"  Should match image aspect: {image_aspect:.4f}")
    print(f"  Display aspect ratio preserved: {'✓' if abs(scaled_display_aspect - image_aspect) < 0.001 else '✗'}")
    print()
    
    # Calculate padding (centered)
    pad_x = (terminal_cols - scaled_w_pixels) // 2
    pad_y = (terminal_rows - scaled_h_pixels) // 2
    
    print("Padding (centered):")
    print(f"  pad_x = (terminal_cols - scaled_w_pixels) // 2 = ({terminal_cols} - {scaled_w_pixels}) // 2 = {pad_x}")
    print(f"  pad_y = (terminal_rows - scaled_h_pixels) // 2 = ({terminal_rows} - {scaled_h_pixels}) // 2 = {pad_y}")
    print()
    
    # Verify dimensions fit
    print("Verification:")
    print(f"  scaled_w_pixels ({scaled_w_pixels}) <= terminal_cols ({terminal_cols}): {'✓' if scaled_w_pixels <= terminal_cols else '✗'}")
    print(f"  scaled_h_pixels ({scaled_h_pixels}) <= terminal_rows ({terminal_rows}): {'✓' if scaled_h_pixels <= terminal_rows else '✗'}")
    print(f"  Total width: {scaled_w_pixels + 2 * pad_x} (should be <= {terminal_cols})")
    print(f"  Total height: {scaled_h_pixels + 2 * pad_y} (should be <= {terminal_rows})")
    print()
    
    # Calculate display aspect ratio of scaled image
    scaled_display_aspect = (scaled_w_pixels * font_ratio) / scaled_h_pixels if scaled_h_pixels > 0 else 1.0
    print("Display Aspect Ratios:")
    print(f"  Scaled image display aspect: {scaled_display_aspect:.4f} (({scaled_w_pixels} * {font_ratio}) / {scaled_h_pixels})")
    print(f"  Should match image aspect: {image_aspect:.4f}")
    print(f"  Match: {'✓' if abs(scaled_display_aspect - image_aspect) < 0.001 else '✗'}")
    print()
    
    print("=" * 80)
    print("Summary:")
    print(f"  Scale factor: {scale:.6f}")
    print(f"  Scaled dimensions: {scaled_w_pixels}x{scaled_h_pixels}")
    print(f"  Padding: {pad_x} columns (left/right), {pad_y} rows (top/bottom)")
    print(f"  Final output: {terminal_cols}x{terminal_rows} character grid")
    print("=" * 80)
    
    return {
        'scale': scale,
        'scaled_w': scaled_w_pixels,
        'scaled_h': scaled_h_pixels,
        'pad_x': pad_x,
        'pad_y': pad_y,
        'image_aspect': image_aspect,
        'scaled_display_aspect': scaled_display_aspect
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

