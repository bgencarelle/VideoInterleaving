import os
import cv2

# Read folder locations from folder_locations.txt
with open('folder_locations.txt', 'r') as f:
    folders = f.read().splitlines()

# Get a list of all the files in each folder, sorted alphabetically
folder_files = []
for folder in folders:
    files = [os.path.join(folder, file) for file in os.listdir(folder)]
    files.sort()
    folder_files.append(files)

# Merge the files interleaved alphabetically
merged_files = []
while any(folder_files):
    for i in range(len(folder_files)):
        if folder_files[i]:
            merged_files.append(folder_files[i].pop(0))

# Write the merged file list to sorted.txt
with open('sorted.txt', 'w') as f:
    for file_path in merged_files:
        f.write(f"{file_path}\n")

