import csv
import os
import time
from collections import deque
from threading import Event
from midi_control import process_midi_messages

import cv2
import mido
import numpy as np
import psutil

clock_intervals = deque()
intervals_sum = 0.0
frame_counter = 0
highest_timecode_seen = 0
clock_counter = 0
stop_event = Event()
mtc_values = [0, 0, 0, 0]
note = 0
last_clock_time = None
display_clock_time = True
check_bpm = False
clock_message_count = 0
use_midi_beat_clock = False
previous_time = time.time()
avg_clock_interval = 0.1
mtc_img = np.zeros((150, 600), dtype=np.uint8)
last_mtc_timecode = '00:00:00:00:00'
total_frames = 0
mod_value = 0
note_scaled = 0
mask_control = 1

def select_midi_input():
    available_ports = mido.get_input_names()

    # Check if there are any available ports
    if len(available_ports) == 0:
        print("No MIDI input ports available.")
        return None

    # Check if there is only one port or all ports have the same name
    if len(set(available_ports)) == 1:
        selected_port = available_ports[0]
        print("Only one input port available, defaulting to:")
        print(selected_port)
        return selected_port

    # Prompt user to select a port
    print("Please select a MIDI input port:")
    for i, port in enumerate(available_ports):
        print(f"{i + 1}: {port}")

    while True:
        try:
            selection = input("> ")
            if selection == "":
                selected_port = available_ports[0]
                print(f"Defaulting to first MIDI input port: {selected_port}")
                return selected_port

            selection = int(selection)
            if selection not in range(1, len(available_ports) + 1):
                raise ValueError

            selected_port = available_ports[selection - 1]
            print(f"Selected port: {selected_port}")
            return selected_port

        except ValueError:
            print("Invalid selection. Please enter a number corresponding to a port.")


def select_csv_file():
    csv_dir = 'generatedPngLists'
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    if len(csv_files) == 0:
        print(f"No .csv files found in directory {csv_dir}")
        return None

    if len(csv_files) == 1:
        print("Only one .csv file found, defaulting to:")
        selected_file = os.path.join(csv_dir, csv_files[0])
        print(selected_file)
        return selected_file

    print("Please select a .csv file to use:")
    for i, f in enumerate(csv_files):
        print(f"{i + 1}: {f}")

    while True:
        try:
            selection = int(input("> "))
            if selection not in range(1, len(csv_files) + 1):
                raise ValueError
            break
        except ValueError:
            print("Invalid selection. Please enter a number corresponding to a file.")

    selected_file = os.path.join(csv_dir, csv_files[selection - 1])
    print(f"Selected file: {selected_file}")
    return selected_file


def parse_array_file(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row[1:] for row in csv_reader]

    return png_paths


def calculate_total_frames(hours, minutes, seconds, frames):
    total_frames = (hours * 60 * 60 * 30) + (minutes * 60 * 30) + (seconds * 30) + frames
    return total_frames


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 * 1024):.2f} MB")


def calculate_index(estimate_frame_counter, png_paths, index_mult=4.0, frame_duration=8.6326, smoothing_factor=0.1):
    effective_length = len(png_paths)
    progress = (estimate_frame_counter / (frame_duration / index_mult)) % (effective_length * 2)

    if int(progress) <= effective_length:
        index = int(progress)
    else:
        index = int(effective_length * 2 - progress)

    # Apply the smoothing factor to the index calculation
    smoothed_index = int(index * (1 - smoothing_factor) + smoothing_factor * (index + 1))

    # Ensure the index is within the valid range
    index = max(0, min(smoothed_index, effective_length - 1))

    return index

def read_frame(png_paths, index, folder):
    png_file = png_paths[index][folder]
    return cv2.imread(png_file, cv2.IMREAD_UNCHANGED)


def display_png_live(frame, mtc_timecode, estimate_frame_counter, index):
    clean_frame = frame.copy()  # Create a clean copy of the frame
    if display_clock_time:
        timecode_text = f'MTC Timecode: {mtc_timecode}'
        estimate_text = f'Estimate Frame Counter: {estimate_frame_counter}'
        index_text = f'index: {index}'
        cv2.putText(clean_frame, timecode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.putText(clean_frame, estimate_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(clean_frame, index_text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(f'MTC Timecode - PID: {os.getpid()}', clean_frame)


def display_png_filters(index, png_paths, folder, open_cv_filters=None, use_as_top_mask=True, solid_color_mask=True,
                        bg_mask=12, buffer_size=2221):
    # Create a circular buffer (deque) for buff_png if it doesn't exist yet
    if not hasattr(display_png_filters, "buff_png"):
        display_png_filters.buff_png = deque(maxlen=buffer_size)

    effective_length = len(png_paths)

    # Define a function to search the buffer for a frame
    def find_frame_in_buffer(index, folder):
        for buffered_frame in display_png_filters.buff_png:
            if buffered_frame["folder"] == folder and (
                    buffered_frame["index"] == index or buffered_frame["index"] == effective_length - index - 1):
                return buffered_frame["frame"]
        return None

    # Search for the main frame in the buffer
    main_frame = find_frame_in_buffer(index, folder)
    if main_frame is not None:
        return main_frame

    # Search for the background frame in the buffer if necessary
    background_frame = None
    if use_as_top_mask:
        background_frame = find_frame_in_buffer(index, bg_mask)

    # If neither frame is in the buffer, read them from the files
    if main_frame is None:
        png_file = png_paths[index][folder]
        main_frame = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)

    if use_as_top_mask and background_frame is None:
        background_file = png_paths[index][bg_mask]
        background_frame = cv2.imread(background_file, cv2.IMREAD_UNCHANGED)

    # Process the main_frame and background_frame
    bg_color = (30, 32, 30)
    mask_bg_color = (32, 245, 30)

    if main_frame is not None:
        background = np.zeros_like(main_frame[..., 0:3], dtype=np.uint8)
        background[:, :] = bg_color

        if main_frame.shape[2] == 4:
            alpha_channel = main_frame[:, :, 3] / 255.0
            inv_alpha_channel = 1 - alpha_channel

            main_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGRA2BGR)
            main_layer = (main_frame * alpha_channel[..., None]) + (background * inv_alpha_channel[..., None])
            main_layer = main_layer.astype(np.uint8)

        if use_as_top_mask and background_frame is not None:
            if background_frame.shape[2] == 4:
                mask_alpha_channel = background_frame[:, :, 3] / 255.0
                inv_mask_alpha_channel = 1 - mask_alpha_channel
            else:
                mask_alpha_channel = np.ones(main_layer.shape[:2], dtype=np.float32)
                inv_mask_alpha_channel = np.zeros(main_layer.shape[:2], dtype=np.float32)

            if solid_color_mask:
                floating_mask = np.zeros_like(main_layer, dtype=np.uint8)
                floating_mask[:, :] = mask_bg_color
            else:
                floating_mask = cv2.cvtColor(background_frame, cv2.COLOR_BGRA2BGR)
                floating_mask = cv2.resize(floating_mask, (main_frame.shape[1], main_frame.shape[0]))

            main_layer = (main_layer * inv_mask_alpha_channel[..., None]) + (
                    floating_mask * mask_alpha_channel[..., None])
            main_layer = main_layer.astype(np.uint8)

        if open_cv_filters:
            for open_cv_filter in open_cv_filters:
                main_layer = open_cv_filter(main_layer)

        # Add the processed frames to the buffer
        display_png_filters.buff_png.append({"index": index, "folder": folder, "frame": main_layer})

        if use_as_top_mask:
            display_png_filters.buff_png.append({"index": index, "folder": bg_mask, "frame": background_frame})

        return main_layer

def scale_value(value, input_min, input_max, output_min, output_max):
    return ((value - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min


def get_color_image(png_paths):
    # Read the main frame from the files
    bg_color = (30, 32, 30)
    png_file = png_paths[0][0]
    main_frame = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)
    background = np.zeros_like(main_frame[..., 0:3], dtype=np.uint8)
    background[:, :] = bg_color
    return background

def mtc_png_realtime_midi(png_paths,):
    global mod_value, note_scaled, display_clock_time, mask_control
    while not stop_event.is_set():


        note_scaled = note % 12  # this scales the keys to one octave
        index = calculate_index(total_frames, png_paths)
        frame = display_png_filters(index, png_paths, note_scaled, False, True, mask_control)
        if frame is not None:
            display_png_live(frame, mtc_timecode_local, total_frames, index)

        # Break the loop if the 'q' key is pressed
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed in (ord('q'), ord('f'), ord('c'), ord('1')):
            if key_pressed == ord('q'):
                stop_event.set()
            if key_pressed == ord('f'):
                print_memory_usage()
            if key_pressed == ord('c'):
                display_clock_time = not display_clock_time
                print("clock display toggled")
            if key_pressed == ord('1'):
                mask_control = (mask_control + 1) % 5
                print(mask_control)
    # Clean up
    input_port.close()
    cv2.destroyAllWindows()


def main():
    mtc_timecode_local, total_frames, note, mod_value \
        = process_midi_messages(input_port, listening_channels)
    mtc_png_realtime_midi(png_paths, input_port, listening_channels)



if __name__ == "__main__":
    listening_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                          15]  # For example, listen to channels 1, 2, and 3
    midi_input_port = select_midi_input()  # Let user choose MIDI input port
    input_port = mido.open_input(midi_input_port)
    csv_source = select_csv_file()
    png_paths = parse_array_file(csv_source)
    background_color = get_color_image(png_paths)
    main()
