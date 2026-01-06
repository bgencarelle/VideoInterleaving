# Utilities

This directory contains utility scripts for image processing, validation, conversion, and asset preparation for the VideoInterleaving project.

## Table of Contents

- [Validation](#validation)
- [Conversion](#conversion)
- [Processing](#processing)
- [Asset Preparation](#asset-preparation)
- [Archived Scripts](#archived-scripts)

---

## Validation

### `check_webp.py`

Validates WebP files using the same libwebp library path as the main application. Useful for detecting corrupted or invalid WebP files that might cause issues during playback.

**Dependencies:**
- `numpy`
- `ctypes` (standard library)
- `libwebp` system library

**Usage:**
```bash
python check_webp.py
# Interactive mode prompts for directory, workers, and log file options
```

**Features:**
- Parallel validation using ThreadPoolExecutor
- Optional log file for bad files
- Uses the same libwebp loading pattern as the main application

---

### `find_missing.py`

Finds missing files in a numbered sequence (e.g., `image001.png`, `image002.png`, `image005.png` would report `image003.png` and `image004.png` as missing).

**Usage:**
```bash
python find_missing.py
# Interactive mode prompts for folder path
```

**Features:**
- Detects gaps in numbered file sequences
- Handles different file extensions
- Simple, focused utility

---

## Conversion

### `image_resize.py`

Batch converts images to WebP format with resizing using `cwebp`. Preserves directory structure and supports various input formats.

**Dependencies:**
- `cwebp` command-line tool (from libwebp package)
- `pathlib`, `concurrent.futures` (standard library)

**Usage:**
```bash
python image_resize.py
# Interactive mode with prompts for:
# - Parent directory
# - Destination directory (default: {parent_dir}_smol)
# - Target height in pixels
# - WebP quality (0-100, default: 91)
# - Number of worker threads
# - Overwrite existing files
```

**Features:**
- Recursive directory scanning
- Parallel processing with ThreadPoolExecutor
- Optimized cwebp settings (method 6, adaptive filtering, photo preset)
- Progress reporting
- Preserves directory structure

---

### `reencode_for_jps_sbs.py`

Converts WebP/PNG images to JPEG format with side-by-side layout: RGB image on the left, alpha mask on the right. Useful for formats that don't support alpha channels natively.

**Dependencies:**
- `opencv-python` (cv2)
- `numpy`
- `tqdm`
- `concurrent.futures` (standard library)

**Usage:**
```bash
python reencode_for_jps_sbs.py
# Interactive mode prompts for:
# - Source folder path
# - Output folder path (default: {input_dir}_sbs)
# - JPEG quality (1-100, default: 90)
```

**Features:**
- Handles grayscale, BGR, and BGRA images
- Generates alpha mask if missing
- Parallel processing with ProcessPoolExecutor
- Progress bar with tqdm
- Preserves directory structure

---

### `batch.py`

Batch compresses WebP files using `cwebp` with ICC profile preservation. Logs folders containing files without ICC profiles.

**Dependencies:**
- `cwebp` and `webpmux` command-line tools
- `concurrent.futures`, `threading` (standard library)

**Usage:**
```bash
python batch.py
# Interactive mode prompts for source folder path
# Output: {input_dir}_webp90
# Log file: missing_icc_log.txt
```

**Features:**
- Preserves ICC profiles when present
- Logs folders with missing ICC profiles
- Parallel processing with ThreadPoolExecutor
- Uses optimized cwebp settings (quality 90, method 6, sharp_yuv)

---

## Processing

### `whiteBalance.py`

Applies white balance correction to images using a reference image (30th image or last if fewer than 30) from each folder. Outputs lossless WebP files.

**Dependencies:**
- `opencv-python` (cv2)
- `numpy`
- `PIL` (Pillow)
- `concurrent.futures` (standard library)

**Usage:**
```bash
python whiteBalance.py
# Interactive mode prompts for input folder path
# Output: {input_dir}_wb
```

**Features:**
- Per-folder white reference calculation
- White patch detection with Gaussian blur
- Dampened gain correction (50% of computed correction, max 1.2x)
- Fallback to 95th percentile if no white patch found
- Parallel processing with ThreadPoolExecutor
- Preserves directory structure
- Lossless WebP output

---

### `turntoblack.py`

Restores perceptual appearance of images whose alpha masks erased dark RGB data by compositing over black and setting full opacity.

**Dependencies:**
- `PIL` (Pillow)
- `numpy`
- `concurrent.futures` (standard library)

**Usage:**
```bash
# Interactive mode
python turntoblack.py

# Command-line mode
python turntoblack.py /path/to/folder [--dry-run]

# Multiple folders
python turntoblack.py /path/to/folder1 /path/to/folder2
```

**Features:**
- Supports both interactive and command-line modes
- Dry-run mode for testing
- Parallel processing with ThreadPoolExecutor
- Handles PNG and WebP formats
- Lossless WebP output when applicable

---

### `mergeImagesToMovie.py`

Composites images based on CSV files (main image + float image) and saves as PNG. Uses alpha compositing to merge images.

**Dependencies:**
- `PIL` (Pillow)
- `tqdm`
- `argparse`, `logging` (standard library)

**Usage:**
```bash
python mergeImagesToMovie.py [options]

Options:
  -p, --pattern PATTERN     CSV file pattern (default: "*.csv")
  -o, --output DIR          Output folder (default: "merged_images_png")
  -c, --compression LEVEL   PNG compression (0-9, default: 1)
  -l, --log LEVEL           Logging level (default: INFO)
  -n, --name-prefix PREFIX  Output filename prefix (default: "NAME_")
  --padding N                Number of digits for padding (default: 6)
  --workers N                Number of parallel workers
```

**Features:**
- Full argparse implementation
- Configurable logging
- Atomic file writes (temp file + rename)
- Retry mechanism (max 3 attempts)
- Parallel processing with ProcessPoolExecutor
- Progress bar with tqdm
- CSV format: Absolute Index, Main Image Path, Float Image Path

---

## Asset Preparation

### `bake_assets.py`

Packs image folders into single `.npy` memory-mapped files for instant seeking. Resizes images to the target resolution (from `settings.py`) and stores as RGBA arrays.

**Dependencies:**
- `numpy`
- `PIL` (Pillow)
- `settings` module (from parent directory)
- `concurrent.futures` (standard library)

**Usage:**
```bash
python bake_assets.py
# Interactive mode prompts for source root directory
# Output: {input_dir}_slab
```

**Features:**
- Memory-mapped NumPy arrays for efficient access
- Pre-allocated files for instant creation
- Parallel processing with ProcessPoolExecutor
- Preserves directory structure (one `frames.npy` per folder)
- Uses resolution from `settings.HEADLESS_RES` (default: 640x480)
- Nearest-neighbor resize for speed (can be changed to BILINEAR for quality)

**Output Format:**
- Shape: `(frames, height, width, 4)` where 4 = RGBA channels
- Dtype: `uint8`
- File: `frames.npy` in each processed folder

---

## Archived Scripts

The following scripts have been moved to `utilities/archive/` as they are redundant, unused, superseded, or have been replaced by better implementations:

- `ascii_tester.py` - ASCII conversion testing (superseded)
- `background_removal.py` - Background removal utility
- `check_webp_interactive.sh` - Shell script version of check_webp.py
- `generate_transparent_webp.py` - WebP transparency generation
- `MakeAmatte.py` - Matte generation
- `mattePlusWhiteBalance.py` - Combined matte and white balance
- `merge_2_images.py` - Simple image merging (superseded by mergeImagesToMovie.py)
- `overlay.py` - Image overlay utility
- `stripPixels.py` - Pixel stripping utility
- `test_alpha.py` - Alpha channel testing
- `webp_file_compare.py` - WebP file comparison
- `webpTester.py` - WebP testing utility

---

## General Notes

### Common Patterns

Most scripts follow these patterns:
- **Parallel Processing**: Use `ThreadPoolExecutor` for I/O-bound tasks, `ProcessPoolExecutor` for CPU-bound tasks
- **Progress Reporting**: Use `tqdm` for progress bars where appropriate
- **Error Handling**: Try/except blocks with meaningful error messages
- **Path Handling**: Mix of `os.path` and `pathlib.Path` (modernization in progress)

### Dependencies

Common dependencies across utilities:
- `numpy` - Array operations
- `opencv-python` (cv2) - Image processing
- `PIL` (Pillow) - Image I/O
- `tqdm` - Progress bars
- `concurrent.futures` - Parallel processing

### System Requirements

Some scripts require system packages:
- `cwebp`, `webpmux` - From `libwebp` package (install via `apt install webp` or equivalent)
- `libwebp` shared library - For `check_webp.py`

---

## Modernization Status

The following scripts are being modernized with:
- `argparse` for command-line arguments
- `logging` module for output
- Type hints for better code clarity
- Improved error handling
- Consistent use of `pathlib.Path`

**Modernized:**
- ✅ `image_resize.py` - Full argparse implementation
- ✅ `mergeImagesToMovie.py` - Full argparse and logging

**In Progress:**
- 🔄 `check_webp.py`
- 🔄 `reencode_for_jps_sbs.py`
- 🔄 `whiteBalance.py`
- 🔄 `bake_assets.py`
- 🔄 `batch.py`
- 🔄 `find_missing.py`
- 🔄 `turntoblack.py`



