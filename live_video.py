import random
import cv2
import numpy as np

# Read the list of PNG file names from the text file
with open("sorted.txt", "r") as file:
    png_files = [line.strip() for line in file]

# Prompt the user for the number of frames and the frames per second
try:
    num_frames = int(input(f"Enter the number of frames for the video (default {len(png_files)}): "))
except ValueError:
    num_frames = len(png_files)

try:
    fps = int(input("Enter the initial frames per second (default 120): "))
except ValueError:
    fps = 10

# Calculate and display the video length
video_length = num_frames / fps
print(f"The video will be {video_length:.2f} seconds long.")

# Calculate the wait time between frames in milliseconds
wait_time = int(1000 / fps)

fixed_width = 1080
fixed_height = 1440

frame_number = 4000
while True:
    # Determine which PNG to use (repeat list if necessary)
    png_index = frame_number % len(png_files)
    print(png_files[png_index])
    # Read the PNG image and resize to fixed dimensions
    image = cv2.imread(png_files[png_index], cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (fixed_width, fixed_height))

    # Check if the image has an alpha channel; if not, create one with full opacity
    if image.shape[2] == 3:
        alpha_channel = np.full((fixed_height, fixed_width), 255, dtype=np.uint8)
        image = cv2.merge((image, alpha_channel))
    else:
        alpha_channel = image[:, :, 3]

    bgr_image = image[:, :, :3]

    # Create a green background with the fixed dimensions
    green_bg = np.zeros((fixed_height, fixed_width, 3), dtype=np.uint8)
    green_bg[:] = (0, 255, 0)

    # Composite the image with the green background using the alpha channel
    alpha = alpha_channel.astype(float) / 255.0
    alpha_rgb = cv2.merge((alpha, alpha, alpha))
    frame = green_bg.astype(float) * (1.0 - alpha_rgb) + bgr_image.astype(float) * alpha_rgb

    # Display the frame
    cv2.imshow('Real-time Video', frame.astype(np.uint8))

    # Listen for keypresses
    key = cv2.waitKey(wait_time) & 0xFF

    # Exit the loop if the user presses the 'q' key
    if key == ord('q'):
        break

    # Increase FPS if the user presses the '+' key
    elif key == ord('+'):
        fps += 1
        wait_time = max(1, int(1000 / fps))

    # Decrease FPS if the user presses the '-' key
    elif key == ord('-'):
        fps -= 1
        fps = max(1, fps)
        wait_time = int(1000 / fps)

    #frame_number += int(random.randrange(-2, 7))
    frame_number += 1

# Close the window and release resources
cv2.destroyAllWindows()
