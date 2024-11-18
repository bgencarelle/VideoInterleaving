import os
import sys
from PIL import Image
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

def collect_image_files(folders, supported_extensions):
    """
    Collect all image files from the given folders and their subdirectories.
    Returns a dictionary mapping directory paths to lists of image file paths.
    """
    directory_files = {}
    for folder in folders:
        for root, _, files in os.walk(folder):
            # Collect files that match the supported extensions
            image_files = [os.path.join(root, file) for file in files if file.lower().endswith(supported_extensions)]
            if image_files:
                directory_files[root] = image_files
    return directory_files

def get_output_path(input_path, source_root, output_root):
    """
    Generate the output path by replacing the source_root with output_root while keeping the relative path.
    The file name remains the same, but the extension is changed to .webp
    """
    relative_path = os.path.relpath(input_path, source_root)
    output_path = os.path.join(output_root, relative_path)
    output_path = os.path.splitext(output_path)[0] + ".webp"
    return output_path

def check_image_properties(image_path):
    """
    Check if the image has alpha and has the correct size (810x1080).
    Returns a tuple (has_alpha, is_correct_size).
    """
    try:
        with Image.open(image_path) as img:
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and 'transparency' in img.info)
            is_correct_size = img.size == (810, 1080)
            return has_alpha, is_correct_size
    except Exception as e:
        print(f"Failed to check properties of {image_path}: {e}")
        return False, False

def determine_folders_to_process(directory_files):
    """
    Determine which folders need processing by checking the first image in each folder.
    Returns a list of tuples containing the folder path and a list of missing parameters.
    """
    folders_to_process = []
    for directory, image_files in directory_files.items():
        first_image = image_files[0]
        # Check WebP format based on file extension
        is_webp = first_image.lower().endswith('.webp')
        # Check other properties using image properties
        has_alpha, is_correct_size = check_image_properties(first_image)
        missing = []
        if not is_webp:
            missing.append("WebP format")
        if not has_alpha:
            missing.append("Alpha channel")
        if not is_correct_size:
            missing.append("Size 810x1080")
        if missing:
            folders_to_process.append((directory, missing))
        else:
            print(f"Skipping directory (already optimized): {directory}")
    return folders_to_process

def resize_image(image, size=(810, 1080)):
    """
    Resize the image to the exact size without maintaining aspect ratio.
    This may distort the image.
    """
    resized_image = image.resize(size)
    return resized_image

def process_image(task, size=(810, 1080), quality=95):
    """
    Process a single image:
    - Add alpha channel if required
    - Resize to the specified size without maintaining aspect ratio
    - Save as WebP with the specified quality
    """
    input_path, output_path, operation = task
    try:
        if os.path.exists(output_path):
            return  # Skip processing if output already exists

        # Load the image
        with Image.open(input_path) as img:
            # Convert image to RGBA if it's not already in a mode that supports alpha
            if img.mode not in ("RGBA", "LA"):
                img = img.convert("RGBA")

            # Add alpha channel if needed
            if operation in ('add_alpha', 'add_alpha_and_resize'):
                if not (img.mode == "RGBA" or (img.mode == "P" and 'transparency' in img.info)):
                    # Create a new image with an alpha layer
                    alpha_layer = Image.new('L', img.size, 255)  # Fully opaque alpha
                    img.putalpha(alpha_layer)

            # Resize if needed
            if operation in ('resize', 'add_alpha_and_resize'):
                img = resize_image(img, size)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save as WebP with specified quality
            img.save(output_path, 'WEBP', quality=quality, lossless=False, method=6)

    except Exception as e:
        print(f"Failed to process {input_path}: {e}")

def main():
    print("=== Image Batch Processor ===")
    print("Enter folder paths containing images. Type 'quit' to start processing.\n")

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')

    folders = []
    while True:
        folder = input("Enter folder path (or type 'quit' to proceed): ").strip()
        if folder.lower() == 'quit':
            break
        folder = os.path.abspath(folder)
        if os.path.isdir(folder):
            folders.append(folder)
            print(f"Added folder: {folder}\n")
        else:
            print("Invalid folder path. Please try again.\n")

    if not folders:
        print("No folders provided. Exiting.")
        sys.exit(0)

    print("\nCollecting image files...")
    directory_files = collect_image_files(folders, supported_extensions)

    if not directory_files:
        print("No image files found in the provided folders. Exiting.")
        sys.exit(0)

    print(f"Total directories with images to evaluate: {len(directory_files)}\n")

    # Determine which folders need processing
    folders_to_process = determine_folders_to_process(directory_files)

    if not folders_to_process:
        print("All folders meet the specified conditions. No processing needed.")
        sys.exit(0)

    print(f"\nTotal folders to process: {len(folders_to_process)}\n")
    for idx, (folder, missing) in enumerate(folders_to_process, 1):
        missing_str = ", ".join(missing)
        print(f"{idx}. {folder}")
        print(f"   Missing: {missing_str}")

    # Prompt user for confirmation
    confirmation = input("\nDo you want to proceed with processing these folders? (yes/no): ").strip().lower()
    if confirmation != 'yes':
        print("Processing aborted by the user.")
        sys.exit(0)

    # Prepare tasks
    tasks = []
    for directory, _ in folders_to_process:
        image_files = directory_files[directory]
        # Define the output root directory (e.g., source_dir_converted)
        parent_dir = os.path.dirname(directory)
        source_dir_name = os.path.basename(directory)
        output_root = os.path.join(parent_dir, f"{source_dir_name}_converted")
        for img_path in image_files:
            output_path = get_output_path(img_path, directory, output_root)
            # Determine operation based on individual image properties
            # Check WebP format based on file extension
            is_webp = img_path.lower().endswith('.webp')
            has_alpha, is_correct_size = check_image_properties(img_path)
            if is_webp and has_alpha and is_correct_size:
                continue  # Skip images that already meet conditions
            elif is_correct_size and not has_alpha:
                operation = 'add_alpha'
            elif not is_correct_size and has_alpha:
                operation = 'resize'
            else:
                operation = 'add_alpha_and_resize'
            tasks.append((img_path, output_path, operation))

    if not tasks:
        print("No images need processing based on the provided criteria. Exiting.")
        sys.exit(0)

    print(f"\nTotal images to process: {len(tasks)}\n")

    # Define the number of worker processes
    num_workers = cpu_count()
    print(f"Starting processing with {num_workers} parallel workers...\n")

    start_time = time.time()

    # Initialize the multiprocessing pool using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(process_image, task): task for task in tasks}
        # Use tqdm to display progress
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing Images", unit="image"):
            pass  # Progress is handled by tqdm

    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_speed = len(tasks) / elapsed_time if elapsed_time > 0 else 0

    print(f"\nProcessing complete in {elapsed_time:.2f} seconds!")
    print(f"Average speed: {avg_speed:.2f} images/second")
    print("Converted images are saved in directories with the '_converted' suffix alongside the source directories.")

if __name__ == "__main__":
    main()
