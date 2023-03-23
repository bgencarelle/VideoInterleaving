import os
import glob
import cv2

with open('folder_locations.txt', 'r') as f:
    folders = f.read().splitlines()
png_sequences = {}
frame_rate = 30  # Change this to your desired frame rate

# Scan each folder for PNG files
for folder in folders:
    png_files = glob.glob(os.path.join(folder, '*.png'))
    png_sequences[folder] = sorted(png_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))  # Fix indentation here

# Merge sequences
merged_sequence = []
max_frames = max([len(seq) for seq in png_sequences.values()])

for i in range(max_frames):
    for folder in folders:
        if i < len(png_sequences[folder]):
            merged_sequence.append(png_sequences[folder][i])

# Create the video
output_video = 'output_video.avi'
frame = cv2.imread(merged_sequence[0])
height, width, _ = frame.shape
video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'MJPG'), frame_rate, (width, height))

for png in merged_sequence:
    frame = cv2.imread(png)
    video.write(frame)

video.release()
