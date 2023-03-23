import os
import glob
from extract_number import extract_number

def read_folders(folders):
    all_sorted_frames = []
    for folder in folders:
        folder_frames = glob.glob(os.path.join(folder, '*.png'))
        folder_frames.sort(key=lambda x: extract_number(os.path.splitext(os.path.basename(x))[0]))
        all_sorted_frames.append(folder_frames)

    frame_list = []
    for frame_tuple in zip(*all_sorted_frames):
        frame_list.extend(frame_tuple)

    return frame_list
