#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Common still-image extensions cwebp can usually handle
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".jpe",
    ".tif", ".tiff", ".bmp", ".gif",
    ".webp"
}


def ensure_cwebp_available() -> None:
    """Verify that cwebp is installed and on PATH."""
    if shutil.which("cwebp") is None:
        print("Error: 'cwebp' not found on PATH.")
        print("Install libwebp tools, e.g.:  sudo apt install webp")
        sys.exit(1)


def find_images(src_dir: Path) -> list[Path]:
    """Return a list of image file paths under src_dir."""
    files: list[Path] = []
    for root, _, names in os.walk(src_dir):
        root_path = Path(root)
        for name in names:
            ext = Path(name).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                files.append(root_path / name)
    return files


def build_dest_path(src_file: Path, src_root: Path, dest_root: Path) -> Path:
    """
    Build destination .webp path, preserving directory structure
    relative to src_root and changing extension to .webp.
    """
    rel = src_file.relative_to(src_root)
    rel_no_ext = rel.with_suffix("")  # strip original extension
    return dest_root / rel_no_ext.with_suffix(".webp")


def run_cwebp(
    src: Path,
    dest: Path,
    height: int,
    quality: int,
    overwrite: bool,
) -> None:
    """Encode a single image to WebP via cwebp with resizing (optimized settings)."""
    # Ensure destination directory exists
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        # Skip existing file
        return

    # cwebp command:
    #  -q <quality>        : WebP quality (kept as given)
    #  -m 6                : slowest, best compression
    #  -mt                 : multi-threaded encoding
    #  -preset photo       : good general preset for photographic images
    #  -af                 : adaptive filtering (often better quality/size tradeoff)
    #  -resize 0 <height>  : preserve aspect ratio, set height
    #  -metadata none      : drop EXIF/ICC/XMP to save bytes (no visual impact)
    #  -quiet              : no stdout spam
    cmd = [
        "cwebp",
        "-quiet",
        "-mt",
        "-preset", "photo",
        "-m", "6",
        "-af",
        "-q", str(quality),
        "-metadata", "none",
    ]

    if height > 0:
        cmd += ["-resize", "0", str(height)]

    cmd += [str(src), "-o", str(dest)]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"cwebp failed for {src} -> {dest}: {err}")


def process_images(
    src_root: Path,
    dest_root: Path,
    height: int,
    quality: int,
    workers: int,
    overwrite: bool,
) -> None:
    """Dispatch parallel cwebp conversions with progress reporting."""
    files = find_images(src_root)
    total = len(files)

    if total == 0:
        print(f"No images found in {src_root}")
        return

    print(f"\nFound {total} image(s) under {src_root}")
    print(f"Output directory : {dest_root}")
    print(f"Target height    : {height}px")
    print(f"WebP quality     : {quality}")
    print(f"Workers          : {workers}")
    print(f"Overwrite files  : {'yes' if overwrite else 'no'}\n")

    completed = 0
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for src in files:
            dest = build_dest_path(src, src_root, dest_root)
            fut = executor.submit(run_cwebp, src, dest, height, quality, overwrite)
            future_map[fut] = (src, dest)

        for fut in as_completed(future_map):
            src, dest = future_map[fut]
            try:
                fut.result()
            except Exception as e:
                failed.append(f"{src} -> {dest}: {e}")
            finally:
                completed += 1
                print(f"Processed {completed}/{total} images", end="\r", flush=True)

    print()  # newline after progress

    if failed:
        print(f"\nCompleted with {len(failed)} failures:")
        for line in failed:
            print("  -", line)
    else:
        print("\nAll images processed successfully.")


def default_dest_dir(parent_dir: Path) -> Path:
    """Mimic the old *_smol naming convention."""
    return parent_dir.with_name(parent_dir.name.rstrip(os.sep) + "_smol")


def prompt_path(prompt: str, default: Path | None = None) -> Path:
    while True:
        raw = input(
            f"{prompt}"
            + (f" [default: {default}]" if default is not None else "")
            + ": "
        ).strip()
        if not raw and default is not None:
            return default
        p = Path(raw).expanduser()
        if p.exists() or default is None:
            return p
        print(f"Path '{p}' does not exist, try again.")


def prompt_int(prompt: str, min_value: int | None = None, default: int | None = None) -> int:
    while True:
        base = f"{prompt}"
        if default is not None:
            base += f" [default: {default}]"
        base += ": "
        raw = input(base).strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
            if min_value is not None and value < min_value:
                print(f"Value must be >= {min_value}")
                continue
            return value
        except ValueError:
            print("Please enter an integer.")


def prompt_yes_no(prompt: str, default: bool | None = None) -> bool:
    while True:
        if default is True:
            suffix = " [Y/n]: "
        elif default is False:
            suffix = " [y/N]: "
        else:
            suffix = " [y/n]: "
        raw = input(prompt + suffix).strip().lower()
        if not raw and default is not None:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer y or n.")


def main() -> None:
    print("=== Batch WebP Resizer (cwebp, interactive) ===")
    print("This will recursively convert images to WebP (quality 91 by default).")
    ensure_cwebp_available()

    # Parent directory
    parent_dir_raw = input("Enter the parent directory [default: current directory]: ").strip()
    if not parent_dir_raw:
        parent_dir = Path(".").resolve()
    else:
        parent_dir = Path(parent_dir_raw).expanduser().resolve()

    if not parent_dir.exists() or not parent_dir.is_dir():
        print(f"Error: {parent_dir} does not exist or is not a directory.")
        sys.exit(1)

    # Destination directory
    default_dest = default_dest_dir(parent_dir)
    dest_raw = input(f"Enter destination directory [default: {default_dest}]: ").strip()
    if not dest_raw:
        dest_dir = default_dest.resolve()
    else:
        dest_dir = Path(dest_raw).expanduser().resolve()

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Height (required, > 0)
    height = prompt_int("Enter target height in pixels", min_value=1)

    # Quality (default 91)
    quality = prompt_int("Enter WebP quality (0–100)", min_value=0, default=91)

    # Workers (default = CPU count or 4)
    default_workers = os.cpu_count() or 4
    workers = prompt_int("Enter number of worker threads", min_value=1, default=default_workers)

    # Overwrite?
    overwrite = prompt_yes_no("Overwrite existing .webp files?", default=False)

    process_images(
        src_root=parent_dir,
        dest_root=dest_dir,
        height=height,
        quality=quality,
        workers=workers,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()
