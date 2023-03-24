import cv2
import numpy as np

# Read the list of PNG file names from the text file
with open("sorted.txt", "r") as file:
    png_files = [line.strip() for line in file]

# Prompt the user for the number of frames and the frames per second
print("the default length is " +str(len(png_files)) +" frames long")
num_frames_input = input("Enter the number of frames for the video: ")
if num_frames_input.isdigit() and int(num_frames_input) < len(png_files):
    num_frames = int(num_frames_input)
else:
    num_frames = len(png_files)

fps_input = input("Enter the frames per second (default is 30): ")
if fps_input.isdigit():
    fps = int(fps_input)
else:
    fps = 30  # default to 30 FPS

# Calculate and display the video length
video_length = num_frames / fps
print(f"The video will be {video_length:.2f} seconds long.")

# Read the first image to get dimensions and create a VideoWriter object
first_image = cv2.imread(png_files[0], cv2.IMREAD_UNCHANGED)
height, width, _ = first_image.shape
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

# Iterate through the list of PNGs, adding each to the video as a frame
for frame_number in range(num_frames):
    # Determine which PNG to use (repeat list if necessary)
    png_index = frame_number % len(png_files)
    print("frame number:" +str(png_index))
    print(png_files[png_index])
    # Read the PNG image and resize if needed
    image = cv2.imread(png_files[png_index], cv2.IMREAD_UNCHANGED)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height))

    # Check if the image has an alpha channel; if not, create one with full opacity
    if image.shape[2] == 3:
        alpha_channel = np.full((height, width), 255, dtype=np.uint8)
        image = cv2.merge((image, alpha_channel))
    else:
        alpha_channel = image[:, :, 3]

    bgr_image = image[:, :, :3]

    # Create a green background with the same dimensions as the input images
    green_bg = np.zeros((height, width, 3), dtype=np.uint8)
    green_bg[:] = (0, 255, 0)

    # Composite the image with the green background using the alpha channel
    alpha = alpha_channel.astype(float) / 255.0
    alpha_rgb = cv2.merge((alpha, alpha, alpha))
    frame = green_bg.astype(float) * (1.0 - alpha_rgb) + bgr_image.astype(float) * alpha_rgb

    # Add the frame to the video
    out.write(frame.astype(np.uint8))

# Release the video file
out.release()
print("Video has been successfully created.")
