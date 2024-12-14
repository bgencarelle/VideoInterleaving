import csv
import os
import glob
import shutil
from datetime import datetime
import cv2
from multiprocessing import Pool, cpu_count
import subprocess

# Constants
FRAME_LIMIT = None  # Set to an integer (e.g., 30) for testing or None to process all frames
TEMP_MERGED_DIR = "temp_merged_frames"
MERGED_FRAME_PREFIX = "frame"
MERGED_FRAME_EXTENSION = ".png"  # Save merged images as PNG to preserve transparency
MAIN_FLOAT_CSV_PATTERN = "main_float_*.csv"
FRAMERATE = 30  # Default framerate, can be modified as needed

def find_csv_files(pattern=MAIN_FLOAT_CSV_PATTERN):
    """
    Find all CSV files matching the given pattern in the current directory.
    """
    return glob.glob(pattern)

def read_csv(file_path, frame_limit=None):
    """
    Read the CSV file and return a list of tuples containing:
    (Absolute Index, Main Image Path, Float Image Path)
    """
    frames = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) < 3:
                print(f"Skipping incomplete row in {file_path}: {row}")
                continue
            abs_index, main_image, float_image = row[:3]
            frames.append((abs_index, main_image, float_image))
            if frame_limit and len(frames) >= frame_limit:
                break
    return frames

def ensure_directory(path):
    """
    Ensure that the given directory exists. If not, create it.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def merge_images_opencv(args):
    """
    Merge two images using OpenCV with alpha blending and save the result.
    Mimics OpenGL's glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA).
    """
    idx, main_img_path, float_img_path, output_dir = args
    merged_frame_name = f"{MERGED_FRAME_PREFIX}{idx:04d}{MERGED_FRAME_EXTENSION}"
    merged_frame_path = os.path.join(output_dir, merged_frame_name)

    try:
        # Read main and float images with alpha channel
        main_img = cv2.imread(main_img_path, cv2.IMREAD_UNCHANGED)
        float_img = cv2.imread(float_img_path, cv2.IMREAD_UNCHANGED)

        if main_img is None:
            print(f"Main image not found or unreadable: {main_img_path}. Skipping frame {idx}.")
            return False
        if float_img is None:
            print(f"Float image not found or unreadable: {float_img_path}. Skipping frame {idx}.")
            return False

        # Ensure both images have alpha channels
        if main_img.shape[2] == 3:
            main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2BGRA)
        if float_img.shape[2] == 3:
            float_img = cv2.cvtColor(float_img, cv2.COLOR_BGR2BGRA)

        # Resize float image to match main image size if necessary
        if main_img.shape[:2] != float_img.shape[:2]:
            float_img = cv2.resize(float_img, (main_img.shape[1], main_img.shape[0]), interpolation=cv2.INTER_AREA)

        # Extract BGR and Alpha channels
        main_bgr = main_img[:, :, :3].astype(float)
        main_alpha = main_img[:, :, 3].astype(float) / 255.0

        float_bgr = float_img[:, :, :3].astype(float)
        float_alpha = float_img[:, :, 3].astype(float) / 255.0

        # Perform alpha blending
        blended_bgr = (float_bgr * float_alpha[:, :, None]) + (main_bgr * (1 - float_alpha[:, :, None]))
        blended_alpha = (float_alpha + main_alpha * (1 - float_alpha))

        # Combine BGR and Alpha channels
        blended_bgr = blended_bgr.astype('uint8')
        blended_alpha = (blended_alpha * 255).astype('uint8')
        blended_img = cv2.merge((blended_bgr, blended_alpha))

        # Save the blended image as PNG
        cv2.imwrite(merged_frame_path, blended_img)
        print(f"Merged frame {idx}: {merged_frame_name}")
        return True

    except Exception as e:
        print(f"Error merging frame {idx}: {e}")
        return False

def compile_video_ffmpeg(frames_dir, output_video, framerate=FRAMERATE):
    """
    Compile merged frames into a video using FFmpeg.
    Assumes frames are named sequentially as frame0000.png, frame0001.png, etc.
    """
    # FFmpeg requires a consistent naming pattern. Ensure frames are zero-padded.
    input_pattern = os.path.join(frames_dir, f"{MERGED_FRAME_PREFIX}%04d{MERGED_FRAME_EXTENSION}")  # e.g., frame0000.png

    command = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-preset", "fast",  # Use 'fast' preset for quicker encoding
        "-pix_fmt", "yuv420p",  # Ensure compatibility
        output_video
    ]

    try:
        print(f"Compiling video: {output_video}")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video created: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling video:\nCommand: {' '.join(command)}\nError: {e.stderr.decode()}")

def process_csv_file(csv_file, frame_limit=FRAME_LIMIT, framerate=FRAMERATE):
    """
    Process a single CSV file: merge images and compile them into a video.
    """
    print(f"\nProcessing CSV: {csv_file}")

    # Read frames from CSV
    frames = read_csv(csv_file, frame_limit)
    print(f"Number of frames to process: {len(frames)}")

    if not frames:
        print(f"No valid frames found in {csv_file}. Skipping.")
        return

    # Prepare temporary directory for merged frames
    temp_dir = TEMP_MERGED_DIR
    ensure_directory(temp_dir)

    # Prepare arguments for multiprocessing
    merge_args = [
        (idx, main_img, float_img, temp_dir)
        for idx, (abs_index, main_img, float_img) in enumerate(frames)
    ]

    # Use multiprocessing Pool to merge images in parallel
    cpu_cores = cpu_count()
    print(f"Starting image merging with {cpu_cores} parallel processes...")
    with Pool(processes=cpu_cores) as pool:
        results = pool.map(merge_images_opencv, merge_args)

    successful_merges = sum(results)
    print(f"Successfully merged {successful_merges}/{len(frames)} frames.")

    if successful_merges == 0:
        print(f"No frames were successfully merged for {csv_file}. Skipping video compilation.")
        shutil.rmtree(temp_dir)
        return

    # Compile merged frames into a video
    csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
    output_video = f"{csv_basename}.mp4"
    compile_video_ffmpeg(temp_dir, output_video, framerate=framerate)

    # Clean up temporary frames
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary frames in {temp_dir}")

def main():
    """
    Main function to process all relevant CSV files in the current directory.
    """
    start_time = datetime.now()
    print(f"Script started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Find all relevant CSV files
    csv_files = find_csv_files()
    if not csv_files:
        print(f"No CSV files found matching the pattern '{MAIN_FLOAT_CSV_PATTERN}'. Exiting.")
        return

    print(f"Found {len(csv_files)} CSV file(s) to process.")

    # Optional: Ask the user for the desired framerate
    global FRAMERATE
    while True:
        try:
            user_input = input(f"Enter the desired framerate (default {FRAMERATE} FPS): ").strip()
            if user_input == "":
                break  # Use default
            framerate_input = int(user_input)
            if framerate_input <= 0:
                print("Please enter a positive integer for framerate.")
                continue
            FRAMERATE = framerate_input
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer for framerate.")

    # Process each CSV file
    for csv_file in csv_files:
        process_csv_file(csv_file, frame_limit=FRAME_LIMIT, framerate=FRAMERATE)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nScript completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")

if __name__ == "__main__":
    main()
