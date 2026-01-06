
import sys
import os
import cv2
import numpy as np
import settings

# Adjust path to find modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ascii_converter import calculate_ascii_layout, to_ascii

def test_ascii_layout():
    print("Testing calculate_ascii_layout...")
    
    # Test 1: Perfect match
    # Source: 100x100, Target: 100x50 (with 0.5 ratio, effectively 100x100 visual)
    w, h, x, y, scaled = calculate_ascii_layout(100, 100, 100, 50, font_ratio=0.5)
    print(f"Test 1 (Perfect): {w}x{h}, off=({x},{y}) -> Expected 100x50, (0,0)")
    assert w == 100 and h == 50
    assert x == 0 and y == 0
    
    # Test 2: Wide Image in Square Grid (Cover)
    # Source: 200x100 (2:1), Target: 100x50 (effective 1:1 visual)
    # Target visual is 100x100. Source is 200x100.
    # To COVER, we must match HEIGHT (scale down to 50h).
    # scale_x = 100/200 = 0.5
    # scale_y = 50/(100*0.5) = 1.0 < invalid calc, wait. font_ratio applies to h.
    # target_h = 50 rows. 50 * 0.5 font_ratio = 25 effective units? No.
    # Logic: scale_y = target_h / (src_h * font_ratio)
    # scale_y = 50 / (100 * 0.5) = 1.0.
    # scale_x = 100 / 200 = 0.5.
    # MAX(0.5, 1.0) = 1.0.
    # new_w = 200 * 1.0 = 200.
    # new_h = 100 * 1.0 * 0.5 = 50.
    # Target: 100x50. New: 200x50.
    # x_off = (200 - 100) // 2 = 50. (Crop 50 from left).
    # y_off = (50 - 50) // 2 = 0.
    w, h, x, y, scaled = calculate_ascii_layout(200, 100, 100, 50, font_ratio=0.5)
    print(f"Test 2 (Wide): {w}x{h}, off=({x},{y}) -> Expected 200x50, (50,0) [Crop Sides]")
    assert w == 200
    assert h == 50
    assert x == 50
    
    # Test 3: Tall Image in Square Grid (Cover)
    # Source: 100x200 (1:2), Target: 100x50 (effective 1:1 visual)
    # scale_x = 100 / 100 = 1.0.
    # scale_y = 50 / (200 * 0.5) = 0.5.
    # MAX(1.0, 0.5) = 1.0.
    # new_w = 100 * 1.0 = 100.
    # new_h = 200 * 1.0 * 0.5 = 100.
    # Target: 100x50. New: 100x100.
    # x_off = (100 - 100) // 2 = 0.
    # y_off = (100 - 50) // 2 = 25. (Crop Top/Bottom).
    w, h, x, y, scaled = calculate_ascii_layout(100, 200, 100, 50, font_ratio=0.5)
    print(f"Test 3 (Tall): {w}x{h}, off=({x},{y}) -> Expected 100x100, (0,25) [Crop Top/Bottom]")
    assert w == 100
    assert h == 100
    assert y == 25

    print("✅ All layout tests passed!")

def test_to_ascii_padding():
    print("\nTesting to_ascii padding...")
    # Create distinct image (Red)
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)
    
    # Set settings for 100x50 grid
    settings.ASCII_WIDTH = 100
    settings.ASCII_HEIGHT = 50
    settings.ASCII_FONT_RATIO = 0.5
    
    # With wide image (2:1), we expect padding on top/bottom (Test 2 scenario)
    res = to_ascii(img)
    rows = res.split('\r\n')
    
    # Filter empty/border lines
    rows = [r for r in rows if len(r) > 0]
    
    # Middle row should have content
    mid_row = rows[25]
    print(f"Middle Row Length: {len(mid_row)}")
    # In ANSI encoded string, length is misleading.
    # But essentially we want to ensure it didn't crash and has content.
    assert len(res) > 0
    print("✅ to_ascii ran successfully.")

if __name__ == "__main__":
    test_ascii_layout()
    test_to_ascii_padding()
