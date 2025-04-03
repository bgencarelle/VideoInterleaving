import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Prompt user for source folder path
INPUT_DIR = input("Enter the full path to the source folder: ").strip()
if not os.path.isdir(INPUT_DIR):
    print(f"Error: '{INPUT_DIR}' is not a valid directory.")
    exit(1)

# Define output folder and log file
OUTPUT_DIR = INPUT_DIR + "_webp90"
LOG_FILE = "missing_icc_log.txt"

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

# Thread-safe set and lock for logging per folder
logged_folders = set()
log_lock = threading.Lock()

# Initialize the log file (overwrite previous content)
with open(LOG_FILE, "w") as log_fh:
    log_fh.write("Missing ICC profile logged per folder:\n")

def check_icc(file_path):
    """Returns True if an ICC profile is present, otherwise False."""
    try:
        result = subprocess.run(
            ["webpmux", "-get", "icc", file_path],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking ICC for {file_path}: {e}")
        return False

def compress_webp(input_file, output_file, preserve_icc):
    """Compress the file using cwebp, preserving ICC metadata if available."""
    cmd = BASE_CMD.copy()
    if preserve_icc:
        cmd += METADATA_FLAG
    cmd += [input_file, "-o", output_file]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Compressed: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Compression failed for {input_file}: {e}")

def process_file(root, file):
    """Processes a single file: checks for ICC, logs if missing, and compresses."""
    rel_dir = os.path.relpath(root, INPUT_DIR)
    input_file = os.path.join(root, file)
    out_dir = os.path.join(OUTPUT_DIR, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, file)

    # Check for ICC profile
    has_icc = check_icc(input_file)
    if not has_icc:
        with log_lock:
            if rel_dir not in logged_folders:
                with open(LOG_FILE, "a") as log_fh:
                    log_fh.write(f"Missing ICC in: {rel_dir}\n")
                logged_folders.add(rel_dir)

    # Compress the image (preserve metadata only if ICC is present)
    compress_webp(input_file, output_file, preserve_icc=has_icc)

def main():
    # Gather all .webp files with their root directories
    tasks = []
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(".webp"):
                tasks.append((root, file))

    print(f"Processing {len(tasks)} files...")

    # Process files concurrently
    max_workers = os.cpu_count() * 2  # Adjust worker count if needed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, root, file) for root, file in tasks]
        for future in as_completed(futures):
            try:
                future.result()  # Propagate exceptions, if any
            except Exception as e:
                print(f"Error processing file: {e}")

    print(f"\nProcessing complete. See '{LOG_FILE}' for missing ICC profile logs.")

if __name__ == "__main__":
    main()
