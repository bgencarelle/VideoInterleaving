import os
import glob
import cv2
import numpy as np

# Disable libpng warnings
os.environ['PNG_WARNINGS'] = '0'

# Read folder locations and store them in a list
with open('folder_locations.txt', 'r') as f:
    folders = f.read().splitlines()

# Read frames order and store them in a list
with open('frames_order.txt', 'r') as f:
    frame_list = f.read().splitlines()

# Get user inputs for the number of frames to process and black frames between each frame
max_frames = int(input("Enter the number of frames to process: "))
black_frames_between = int(input("Enter the number of black frames between each frame: "))

# Read the first frame to get dimensions for the video
first_frame = cv2.imread(frame_list[0], cv2.IMREAD_UNCHANGED)
height, width, _ = first_frame.shape

# Create the video writer
output_video = 'output_video.mp4'
video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'avc1'), 30, (width, height), isColor=True)

# Process the frames and create the video
for index, png in enumerate(frame_list[:max_frames]):
    print(f"Processing frame {index + 1}/{max_frames}")

    # Read the frame with transparency
    frame = cv2.imread(png, cv2.IMREAD_UNCHANGED)

    # Convert the frame to BGRA and then to BGR
    bgra = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
    bgr_bg = cv2.cvtColor(bgra, cv2.COLOR_RGBA2BGR)

    # Replace transparent pixels with green
    bgr_bg[bgra[:, :, 3] == 0] = [0, 255, 0]

    # Write the frame to the video
    video.write(bgr_bg)

    # Add black frames
    for _ in range(black_frames_between):
        video.write(np.zeros((height, width, 3), dtype=np.uint8))

# Release the video writer
video.release()
