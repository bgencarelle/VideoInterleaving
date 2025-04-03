import cv2
import numpy as np
from collections import deque
import os # Added for path manipulation

# --- Configuration ---
# Choose the backend for loading .webp files: 'cv2', 'libvips', 'webp'
# Make sure the corresponding library is installed!
WEBP_LOADER_BACKEND = 'webp' # Change this to 'libvips' or 'webp' to test others

# --- Optional: Import backend libraries ---
# It's good practice to check imports early or handle ImportErrors
try:
    if WEBP_LOADER_BACKEND == 'libvips':
        import pyvips
except ImportError:
    print(f"Warning: Cannot import pyvips. 'libvips' backend will not work.")
    if WEBP_LOADER_BACKEND == 'libvips':
        raise # Or fallback: WEBP_LOADER_BACKEND = 'cv2'

try:
    if WEBP_LOADER_BACKEND == 'webp':
        import webp
except ImportError:
    print(f"Warning: Cannot import webp. 'webp' backend will not work.")
    if WEBP_LOADER_BACKEND == 'webp':
        raise # Or fallback: WEBP_LOADER_BACKEND = 'cv2'


# Assuming settings.py contains these paths (replace with actual paths if needed)
# Example placeholder paths:
MAIN_FOLDER_PATH = [] # Placeholder - list of lists/dicts expected by original code
FLOAT_FOLDER_PATH = [] # Placeholder - list of lists/dicts expected by original code
# from settings import MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH


class ImageLoader:
    # Default paths removed from __init__ signature to avoid confusion if set later
    def __init__(self, main_folder_path=None, float_folder_path=None, png_paths_len=0):
        # Use provided paths or fall back to imported defaults
        self.main_folder_path = main_folder_path if main_folder_path is not None else MAIN_FOLDER_PATH
        self.float_folder_path = float_folder_path if float_folder_path is not None else FLOAT_FOLDER_PATH
        self.png_paths_len = png_paths_len
        print(f"ImageLoader initialized. Using '{WEBP_LOADER_BACKEND}' backend for WebP files.")

    def set_paths(self, main_folder_path, float_folder_path):
        self.main_folder_path = main_folder_path
        self.float_folder_path = float_folder_path

    def set_png_paths_len(self, value):
        self.png_paths_len = value

    def read_image(self, image_path):
        """
        Loads an image from the given path, ensuring RGBA NumPy array output.
        Uses the configured backend for .webp files. Uses cv2 for .png.
        """
        if not os.path.exists(image_path):
             raise FileNotFoundError(f"Image file not found: {image_path}")

        file_ext = os.path.splitext(image_path)[1].lower()

        image_np = None

        try:
            if file_ext == '.png':
                # Use cv2 for PNG files
                image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                if image_np is None:
                    raise ValueError(f"Failed to load PNG image with cv2: {image_path}")

                # Ensure RGBA format for consistency
                if len(image_np.shape) == 2:  # Grayscale
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGBA)
                elif image_np.shape[2] == 3:  # BGR
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
                elif image_np.shape[2] == 4:  # BGRA -> RGBA
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGBA)

            elif file_ext == '.webp':
                # Use the configured backend for WebP files
                if WEBP_LOADER_BACKEND == 'cv2':
                    image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    if image_np is None:
                        raise ValueError(f"Failed to load WebP image with cv2: {image_path}")

                    # Ensure RGBA format
                    if len(image_np.shape) == 2:
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGBA)
                    elif image_np.shape[2] == 3: # BGR
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGBA)
                    elif image_np.shape[2] == 4: # BGRA
                        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGBA) # Convert if needed

                elif WEBP_LOADER_BACKEND == 'libvips':
                    if 'pyvips' not in globals():
                         raise ImportError("pyvips library is required but not imported successfully.")
                    try:
                        vips_image = pyvips.Image.new_from_file(image_path)

                        # Ensure 4 bands (RGBA like) - VIPS usually uses RGB(A)
                        if vips_image.bands == 3:
                             # Add an opaque alpha channel
                            vips_image = vips_image.bandjoin(255)
                        elif vips_image.bands == 1:
                             # Assume grayscale, convert to sRGB color space then add alpha
                             # This ensures consistency, adjust if specific grayscale handling needed
                            vips_image = vips_image.colourspace('srgb').bandjoin(255)
                        elif vips_image.bands != 4:
                             raise ValueError(f"Unsupported band count {vips_image.bands} from libvips for {image_path}")

                        # Convert VIPS image to NumPy array
                        image_np = np.ndarray(
                            buffer=vips_image.write_to_memory(),
                            dtype=np.uint8, # Assuming 8-bit depth
                            shape=[vips_image.height, vips_image.width, vips_image.bands]
                        )
                        # Note: VIPS outputs RGB(A). Since we standardized on RGBA, this should be correct.
                        # If downstream code specifically expects BGRA, a cv2.cvtColor might be needed here.

                    except pyvips.Error as e:
                        raise ValueError(f"Failed to load WebP image {image_path} with libvips: {e}") from e


                elif WEBP_LOADER_BACKEND == 'webp':

                    if 'webp' not in globals():
                        raise ImportError("webp library is required but not imported successfully.")

                    try:

                        # --- CHANGE HERE ---

                        # Pass the image_path string directly to imread

                        # The 'webp' library's imread seems to expect the path.

                        # Request RGBA format using pilmode argument.

                        image_np = webp.imread(image_path, pilmode='RGBA')

                        # --- END CHANGE ---

                        if image_np is None:
                            # Check if imread returns None on failure (some libraries do)

                            raise ValueError(f"webp.imread returned None for: {image_path}")


                    except FileNotFoundError:  # webp.imread might raise this too

                        raise FileNotFoundError(f"Image file not found by webp.imread: {image_path}")

                    except Exception as e:  # Catch potential decoding errors or other webp issues

                        # Check if the error is the specific null byte error to give a hint

                        if isinstance(e, ValueError) and 'embedded null byte' in str(e):
                            # This shouldn't happen with the fix, but good for debugging

                            print("DEBUG: Still encountering null byte error, check if image_path is correct.")

                        raise ValueError(f"Failed to load WebP image {image_path} with webp.imread: {e}") from e
                else:
                    raise ValueError(f"Invalid WEBP_LOADER_BACKEND configured: {WEBP_LOADER_BACKEND}")

            else:
                raise ValueError(f"Unsupported image format: {file_ext}")

            if image_np is None:
                 # This case should ideally be caught by specific loader errors, but as a fallback:
                 raise ValueError(f"Image loading resulted in None for: {image_path}")

            return image_np

        except Exception as e:
            # Catch any unexpected error during loading/conversion and provide context
            print(f"Error processing image '{image_path}' using backend '{WEBP_LOADER_BACKEND if file_ext == '.webp' else 'cv2'}': {e}")
            raise # Re-raise the exception after logging


    def load_images(self, index, main_folder, float_folder):
        """
        Loads a pair of images from the stored folder paths using read_image.
        Expects self.main_folder_path[index][main_folder] and
        self.float_folder_path[index][float_folder] to be valid paths.
        """
        # Original logic assumed paths were structured this way. Adjust if needed.
        try:
            main_image_path = self.main_folder_path[index][main_folder]
            float_image_path = self.float_folder_path[index][float_folder]
        except IndexError:
             raise IndexError(f"Index {index} out of range for folder paths.")
        except KeyError as e:
             raise KeyError(f"Folder key '{e}' not found for index {index}.")
        except TypeError:
             # Handle cases where the path structure isn't list[index][key]
             # You might need to adjust how paths are retrieved based on your actual structure
             raise TypeError(f"Path structure is not as expected (list[index][key]) for index {index}. "
                             f"main_folder_path type: {type(self.main_folder_path)}, "
                             f"float_folder_path type: {type(self.float_folder_path)}")


        main_image = self.read_image(main_image_path)
        float_image = self.read_image(float_image_path)
        return main_image, float_image

# --- ImageLoaderBuffer remains the same ---
class ImageLoaderBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add_image_future(self, index, future, png_paths_len):
        # Clamp index based on png_paths_len.
        clamped_index = max(0, min(index, png_paths_len - 1))
        # Store the original index requested along with clamped and future?
        # Original code just stores clamped_index. Keeping it that way.
        self.buffer.append((clamped_index, future))

    def get_future_for_index(self, index):
        # Iterate safely over a copy in case of removal
        for item in list(self.buffer):
            buf_index, future = item
            # Match against the possibly clamped index stored in the buffer
            if buf_index == index:
                self.buffer.remove(item) # Remove the retrieved future
                return future
        return None # No future found for this index


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # This is just a conceptual example.
    # You need to provide actual paths and structure.

    # Example dummy structure (replace with your actual settings/paths)
    # Assume MAIN_FOLDER_PATH = [[{"folderA": "path/img1.webp"}], [{"folderA": "path/img2.png"}]]
    # Assume FLOAT_FOLDER_PATH = [[{"folderB": "path/f_img1.webp"}], [{"folderB": "path/f_img2.png"}]]

    # Create dummy files for testing (replace with your real files)
    os.makedirs("temp_images", exist_ok=True)
    dummy_png_path = "temp_images/dummy.png"
    dummy_webp_path = "temp_images/dummy.webp"

    # Create a simple black PNG using OpenCV
    try:
        dummy_png = np.zeros((10, 10, 4), dtype=np.uint8) # BGRA
        dummy_png[:,:,3] = 255 # Make alpha opaque
        cv2.imwrite(dummy_png_path, dummy_png)

        # Create a simple green WebP using OpenCV (requires cv2 built with webp support)
        dummy_webp = np.zeros((10, 10, 3), dtype=np.uint8) # BGR
        dummy_webp[:,:,1] = 255 # Green channel
        # Quality parameter is important for webp
        cv2.imwrite(dummy_webp_path, dummy_webp, [cv2.IMWRITE_WEBP_QUALITY, 80])

        print(f"Created dummy files: {dummy_png_path}, {dummy_webp_path}")

        # Setup paths for the loader (matching the expected structure)
        test_main_paths = [
            {"main": dummy_webp_path},
            {"main": dummy_png_path}
        ]
        test_float_paths = [
             {"float": dummy_webp_path}, # Using same file for simplicity
             {"float": dummy_png_path}
        ]

        loader = ImageLoader(main_folder_path=test_main_paths,
                             float_folder_path=test_float_paths,
                             png_paths_len=2) # Example length

        print(f"\n--- Testing with WEBP_LOADER_BACKEND = '{WEBP_LOADER_BACKEND}' ---")

        # Test loading index 0 (WebP)
        try:
            print("\nLoading index 0 (WebP)...")
            main_img_0, float_img_0 = loader.load_images(index=0, main_folder="main", float_folder="float")
            print(f"Index 0: Main shape={main_img_0.shape}, dtype={main_img_0.dtype}")
            print(f"Index 0: Float shape={float_img_0.shape}, dtype={float_img_0.dtype}")
            assert main_img_0.shape[2] == 4 # Check for 4 channels (RGBA)
        except Exception as e:
            print(f"Error loading index 0: {e}")

        # Test loading index 1 (PNG)
        try:
            print("\nLoading index 1 (PNG)...")
            main_img_1, float_img_1 = loader.load_images(index=1, main_folder="main", float_folder="float")
            print(f"Index 1: Main shape={main_img_1.shape}, dtype={main_img_1.dtype}")
            print(f"Index 1: Float shape={float_img_1.shape}, dtype={float_img_1.dtype}")
            assert main_img_1.shape[2] == 4 # Check for 4 channels (RGBA)
        except Exception as e:
            print(f"Error loading index 1: {e}")

        # Test unsupported format (create a dummy .txt)
        try:
            print("\nTesting unsupported format (.txt)...")
            dummy_txt_path = "temp_images/dummy.txt"
            with open(dummy_txt_path, "w") as f: f.write("test")
            loader.read_image(dummy_txt_path)
        except ValueError as e:
            print(f"Correctly caught error for .txt: {e}")
        except Exception as e:
            print(f"Unexpected error for .txt: {e}")

        # Test non-existent file
        try:
            print("\nTesting non-existent file...")
            loader.read_image("temp_images/non_existent.png")
        except FileNotFoundError as e:
             print(f"Correctly caught error for non-existent file: {e}")
        except Exception as e:
            print(f"Unexpected error for non-existent file: {e}")


    finally:
        # Clean up dummy files
        import shutil
        if os.path.exists("temp_images"):
             print("\nCleaning up temp_images directory...")
             shutil.rmtree("temp_images")