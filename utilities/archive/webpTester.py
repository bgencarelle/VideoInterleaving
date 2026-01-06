import time
import pyvips
import os
from PIL import Image  # Import Pillow
import cv2  # Import OpenCV
import numpy as np
import webp  # Import the webp library
from OpenGL.GL import *
from OpenGL.GLUT import *

# Global variables to store texture and image data
texture_id = 0
image_width = 0
image_height = 0
test_counts = 100  # Test each loading function 10 times


def load_image_vips(filename):
    """Loads an image using pyvips and converts it to RGBA for OpenGL.
       Ensures proper cleanup after reading."""
    try:
        # Load the image with sequential access for better performance
        image = pyvips.Image.new_from_file(filename, access="sequential")

        # Add an alpha channel if the image doesn't have one
        if not image.hasalpha():
            image = image.addalpha()

        # Convert image to sRGB if it's not already in that colorspace
        if image.interpretation != "srgb":
            image = image.colourspace("srgb")

        # Cast the image data to unsigned char if needed
        if image.format != 'uchar':
            image = image.cast("uchar")

        # Retrieve the image data as a bytes-like object
        data = image.write_to_memory()
        image = None  # Explicitly release the image
        return image and image.bands or 4, image and image.width or 0, image and image.height or 0, data
    except Exception as e:
        print(f"Error loading with libvips: {e}")
        return None, None, None, None


def load_image_pil(filename):
    """Loads an image using PIL (Pillow) and converts it to RGBA for OpenGL.
       Uses a context manager to ensure the image is closed after processing."""
    try:
        with Image.open(filename) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            img_data = img.tobytes("raw", "RGBA")
            return 4, img.width, img.height, img_data
    except Exception as e:
        print(f"Error loading with Pillow: {e}")
        return None, None, None, None


def load_image_cv2(filename):
    """Loads an image using OpenCV and converts it to RGBA for OpenGL."""
    try:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)  # cv2 reads as BGR
        if img is None:
            return None, None, None, None
        if len(img.shape) == 2:  # Grayscale
            img_rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif img.shape[2] == 3:  # RGB
            img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        elif img.shape[2] == 4:  # RGBA
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            return None, None, None, None
        height, width, channels = img_rgba.shape
        return channels, width, height, img_rgba.tobytes()
    except Exception as e:
        print(f"Error loading with OpenCV: {e}")
        return None, None, None, None


def load_image_webp(filename):
    """Loads an image using the webp library and converts it to RGBA for OpenGL."""
    try:
        img = webp.load_image(filename, 'RGBA')
        return 4, img.width, img.height, img.tobytes()
    except Exception as e:
        print(f"Error loading with webp: {e}")
        return None, None, None, None


def load_image_cv2_webp(filename):
    """Loads a WebP image using cv2 and cv2.imdecode, converting it to RGBA for OpenGL."""
    try:
        with open(filename, 'rb') as f:
            webp_data = f.read()
        img = cv2.imdecode(np.frombuffer(webp_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None, None, None
        height, width, channels = img.shape
        if channels == 3:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        elif channels == 4:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            return None, None, None, None
        return 4, width, height, img_rgba.tobytes()
    except Exception as e:
        print(f"Error loading with OpenCV (cv2.imdecode): {e}")
        return None, None, None, None


def display():
    """OpenGL display function."""
    global texture_id, image_width, image_height

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(-1, -1)  # Bottom-left
    glTexCoord2f(1, 0)
    glVertex2f(1, -1)  # Bottom-right
    glTexCoord2f(1, 1)
    glVertex2f(1, 1)  # Top-right
    glTexCoord2f(0, 1)
    glVertex2f(-1, 1)  # Top-left
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glutSwapBuffers()
    if texture_id != 0:  # Only delete if a texture was created
        glDeleteTextures(1, [texture_id])  # Delete texture after display
        texture_id = 0  # reset


def create_texture(channels, width, height, data):
    """Creates an OpenGL texture from image data."""
    global texture_id
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Convert data to a NumPy array if it isn't one already
    if not isinstance(data, np.ndarray):
        data = np.frombuffer(data, dtype=np.uint8)

    if channels == 4:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    elif channels == 3:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
    else:
        print(f"Unsupported number of channels: {channels}")
        return False
    return True


if __name__ == "__main__":
    # Prompt for filename
    webp_image_path = input("Enter the path to your WebP image: ").strip()

    if not os.path.exists(webp_image_path):
        print(f"Error: WebP image not found at {webp_image_path}")
        exit()

    # Initialize OpenGL
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 600)  # Set initial window size
    glutCreateWindow(b"WebP Image Display")
    glutDisplayFunc(display)  # Set the display function
    glEnable(GL_TEXTURE_2D)
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Set background color

    # Load and display the image using each library multiple times
    loaders = {
        "libvips": load_image_vips,
        "Pillow": load_image_pil,
        "OpenCV": load_image_cv2,
        "webp": load_image_webp,
        "OpenCV (cv2.imdecode)": load_image_cv2_webp,
    }

    # Dictionary to store load times for each loader
    load_times = {}

    for name, loader in loaders.items():
        load_times[name] = []
        print("-" * 40)
        print(f"Testing {name} ({test_counts} times)")
        for i in range(test_counts):
            start_time = time.time()
            channels, width, height, data = loader(webp_image_path)
            run_load_time = time.time() - start_time

            if data is not None:
                image_width = width
                image_height = height
                load_times[name].append(run_load_time)
                if create_texture(channels, width, height, data):
                    display_start_time = time.time()
                    glutPostRedisplay()  # Trigger display
                    display_time = time.time() - display_start_time
                    print(f"  Run {i + 1}: Load Time: {run_load_time:.6f} sec, Display Time: {display_time:.6f} sec")
                else:
                    print(f"  Run {i + 1}: {name}: Texture creation failed")
            else:
                print(f"  Run {i + 1}: {name}: Failed to load image")

    # "Who Won" section: Calculate average load times and determine the winner.
    print("\n" + "=" * 40)
    print("Who Won? (Lowest Average Load Time)")
    best_loader = None
    best_time = float("inf")
    for name, times in load_times.items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"  {name}: Average Load Time: {avg_time:.6f} sec")
            if avg_time < best_time:
                best_time = avg_time
                best_loader = name
        else:
            print(f"  {name}: No successful loads.")
    if best_loader:
        print(f"\nWinner: {best_loader} with an average load time of {best_time:.6f} sec")
    else:
        print("\nNo loader succeeded.")

    glutMainLoop()  # Start the GLUT event processing loop
    print("Finished")