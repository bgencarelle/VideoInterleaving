# Comparison: Original (7d3c7db5) vs Current Implementation

## Key Differences

### 1. `ascii_converter.py` - `to_ascii()` Function

#### Original (7d3c7db5):
```python
def to_ascii(frame):
    """
    Converts a frame to ASCII using a 'Cover' (Zoom/Crop) scaling method.
    Optimized to crop the image pixels *before* color grading for efficiency.
    """
    # Simple signature - just takes frame
    
    # COVER scaling method:
    scale_x = max_cols / w
    scale_y = max_rows / (h * font_ratio)
    scale = max(scale_x, scale_y)  # Use LARGER scale to COVER
    
    # Calculate oversized dimensions
    new_w = int(w * scale)
    new_h = int(h * scale * font_ratio)
    
    # Resize to oversized grid
    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Crop center max_cols x max_rows area
    x_off = (new_w - max_cols) // 2
    y_off = (new_h - max_rows) // 2
    frame_cropped = frame_resized[y_off : y_off + max_rows, x_off : x_off + max_cols]
    
    # Process the cropped frame...
```

**Key characteristics:**
- **COVER method**: Scales to fill the terminal, then crops the center
- **No padding**: Always fills one dimension completely, crops the other
- **Simple**: No aspect ratio parameters, no source dimensions tracking
- **font_ratio applied in scale calculation**: `scale_y = max_rows / (h * font_ratio)`

#### Current (after changes):
```python
def to_ascii(frame, source_aspect_ratio=None, mode=None):
    """
    Converts a frame to ASCII by filling one dimension and padding the other.
    Scales image to fill either width or height completely, then pads the other dimension.
    """
    # Complex signature with extra parameters
    
    # FIT scaling method with padding:
    # - Calculates which dimension to fill
    # - Scales to fit, then adds padding
    # - Uses source_aspect_ratio for consistency
    # - Complex aspect ratio calculations
```

**Key characteristics:**
- **FIT method**: Scales to fit within bounds, then adds padding
- **Padding**: Adds black padding on one dimension
- **Complex**: Tracks source dimensions, aspect ratios, etc.
- **font_ratio applied differently**: In display aspect calculations

### 2. `image_display.py` - How `to_ascii()` is Called

#### Original (7d3c7db5):
```python
# In the capture section (line 290-292):
elif is_ascii:
    text_frame = ascii_converter.to_ascii(frame)
    exchange.set_frame(text_frame)

# Where frame comes from:
if has_gl:
    # GPU path: read from framebuffer
    raw = window.fbo.read(components=3)
    w_fbo, h_fbo = window.size
    frame = np.frombuffer(raw, dtype=np.uint8).reshape((h_fbo, w_fbo, 3))
else:
    # CPU path: composite images
    tgt_size = HEADLESS_RES if is_web else None
    frame = renderer.composite_cpu(
        cur_main, cur_float,
        main_is_sbs=cur_m_sbs, float_is_sbs=cur_f_sbs,
        target_size=tgt_size
    )
```

**Key characteristics:**
- **Simple call**: Just `to_ascii(frame)` - no extra parameters
- **Frame source**: Either GPU framebuffer or CPU composite
- **No pre-processing**: Frame goes directly to `to_ascii()`

#### Current (after changes):
```python
# More complex - passes source_aspect_ratio and other parameters
# Has source_image_size tracking
# May have pre-processing steps
```

## Summary

The original implementation was **simpler and used COVER scaling**:
1. Scales image to COVER the terminal (one dimension fills, other is oversized)
2. Crops the center to get exact terminal dimensions
3. No padding needed
4. No source dimension tracking
5. Simple function signature

The current implementation uses **FIT scaling with padding**:
1. Scales image to FIT within terminal bounds
2. Adds padding to center the image
3. Tracks source dimensions for consistency
4. More complex calculations

## What Broke?

The user mentioned "core functionality" broke. Likely issues:
1. The complex aspect ratio calculations may have introduced bugs
2. The padding approach may not match the original COVER behavior
3. The extra parameters may not be passed correctly in all code paths
4. The font_ratio handling may have changed incorrectly

