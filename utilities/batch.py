"""
Batch WebP Compressor

Batch compresses WebP files using cwebp with ICC profile preservation.
Logs folders containing files without ICC profiles.
"""
import argparse
import logging
import os
import sys
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )

# Compression command flags
BASE_CMD = [
    "cwebp",
    "-q", "90",
    "-m", "6",
    "-alpha_q", "100",
    "-sharp_yuv",
    "-preset", "photo",
    "-mt"
]
METADATA_FLAG = ["-metadata", "icc"]

def check_icc(file_path: Path) -> bool:
    """Returns True if an ICC profile is present, otherwise False."""
    try:
        result = subprocess.run(
            ["webpmux", "-get", "icc", str(file_path)],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        logging.debug(f"Error checking ICC for {file_path}: {e}")
        return False

def compress_webp(input_file: Path, output_file: Path, preserve_icc: bool, quality: int) -> bool:
    """Compress the file using cwebp, preserving ICC metadata if available."""
    cmd = BASE_CMD.copy()
    # Replace quality in command
    cmd[cmd.index("-q") + 1] = str(quality)
    if preserve_icc:
        cmd += METADATA_FLAG
    cmd += [str(input_file), "-o", str(output_file)]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logging.debug(f"Compressed: {input_file} -> {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Compression failed for {input_file}: {e}")
        return False

def process_file(
    root: Path,
    file: str,
    input_dir: Path,
    output_dir: Path,
    log_file: Optional[Path],
    logged_folders: set[str],
    log_lock: threading.Lock,
    quality: int
) -> None:
    """Processes a single file: checks for ICC, logs if missing, and compresses."""
    rel_dir = root.relative_to(input_dir)
    input_file = root / file
    out_dir = output_dir / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / file

    # Check for ICC profile
    has_icc = check_icc(input_file)
    if not has_icc and log_file is not None:
        with log_lock:
            rel_dir_str = str(rel_dir)
            if rel_dir_str not in logged_folders:
                with log_file.open("a", encoding="utf-8") as log_fh:
                    log_fh.write(f"Missing ICC in: {rel_dir_str}\n")
                logged_folders.add(rel_dir_str)

    # Compress the image (preserve metadata only if ICC is present)
    compress_webp(input_file, output_file, preserve_icc=has_icc, quality=quality)

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch compress WebP files with ICC profile preservation."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=str,
        required=True,
        help="Source folder path containing WebP files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output folder path (default: {input_dir}_webp90)"
    )
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=90,
        choices=range(1, 101),
        help="WebP quality (1-100, default: 90)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count * 2)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="missing_icc_log.txt",
        help="Path to log file for missing ICC profiles (default: missing_icc_log.txt)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Resolve paths
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        logging.error(f"'{input_dir}' is not a valid directory.")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_webp90"
    
    log_file = Path(args.log_file).expanduser().resolve()
    # Initialize the log file (overwrite previous content)
    log_file.write_text("Missing ICC profile logged per folder:\n", encoding="utf-8")
    
    # Gather all .webp files with their root directories
    tasks: list[tuple[Path, str]] = []
    for root, _, files in os.walk(input_dir):
        root_path = Path(root)
        for file in files:
            if file.lower().endswith(".webp"):
                tasks.append((root_path, file))

    if not tasks:
        logging.warning("No .webp files found.")
        sys.exit(0)
    
    logging.info(f"Processing {len(tasks)} file(s)...")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Quality: {args.quality}")
    logging.info(f"Log file: {log_file}")

    # Process files concurrently
    if args.workers:
        max_workers = args.workers
        if max_workers < 1:
            logging.error("Worker count must be >= 1")
            sys.exit(1)
    else:
        max_workers = (os.cpu_count() or 1) * 2
    
    logging.info(f"Using {max_workers} worker(s)")
    
    # Thread-safe set and lock for logging per folder
    logged_folders: set[str] = set()
    log_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_file,
                root,
                file,
                input_dir,
                output_dir,
                log_file,
                logged_folders,
                log_lock,
                args.quality
            )
            for root, file in tasks
        ]
        for future in as_completed(futures):
            try:
                future.result()  # Propagate exceptions, if any
            except Exception as e:
                logging.error(f"Error processing file: {e}")

    logging.info(f"Processing complete. See '{log_file}' for missing ICC profile logs.")

if __name__ == "__main__":
    main()
