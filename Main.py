from config import max_frames
from read_folders import read_folders
from process_frames import process_frames

with open('folder_locations.txt', 'r') as f:
    folders = f.read().splitlines()

frame_list = read_folders(folders)
frame_list = frame_list[:max_frames]

process_frames(frame_list)
3