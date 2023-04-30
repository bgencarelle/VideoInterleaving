import csv
import os
import time
from collections import deque
from threading import Event

import cv2
import mido
import numpy as np
import psutil

CLOCK_BUFFER_SIZE = 200
FRAME_BUFFER_SIZE = 30
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


def parse_mtc(msg):
    mtc_type = msg.frame_type
    value = msg.frame_value
    return mtc_type, value


def update_mtc_timecode(mtc_type, value, mtc_clock_counter):
    global mtc_values

    if mtc_type == 0:
        mtc_values[3] = (mtc_values[3] & 0xF0) | value
    elif mtc_type == 1:
        mtc_values[3] = (mtc_values[3] & 0x0F) | (value << 4)
    elif mtc_type == 2:
        mtc_values[2] = (mtc_values[2] & 0xF0) | value
    elif mtc_type == 3:
        mtc_values[2] = (mtc_values[2] & 0x0F) | (value << 4)
    elif mtc_type == 4:
        mtc_values[1] = (mtc_values[1] & 0xF0) | value
    elif mtc_type == 5:
        mtc_values[1] = (mtc_values[1] & 0x0F) | (value << 4)
    elif mtc_type == 6:
        mtc_values[0] = (mtc_values[0] & 0xF0) | value
    elif mtc_type == 7:
        mtc_values[0] = (mtc_values[0] & 0x0F) | (value << 4)

    hours = mtc_values[0] & 0x1F
    minutes = mtc_values[1]
    seconds = mtc_values[2]
    frames = mtc_values[3]
    total = f'{hours:02}:{minutes:02}:{seconds:02}:{frames:02}'

    return total


def calculate_total_frames(hours, minutes, seconds, frames):
    total_frames = (hours * 60 * 60 * 30) + (minutes * 60 * 30) + (seconds * 30) + frames
    return total_frames


def process_midi_messages(input_port, channels):
    global clock_counter, mtc_values, frame_counter, last_mtc_timecode, note
    global last_clock_time, clock_message_count, previous_time, intervals_sum
    current_total_frames = 0

    for msg in input_port.iter_pending():
        if msg.type == 'clock':

            if check_bpm is True:
                current_time = time.time()
                if last_clock_time is not None:
                    clock_interval = current_time - last_clock_time

                    if len(clock_intervals) == CLOCK_BUFFER_SIZE:
                        # Remove the oldest interval from the sum
                        intervals_sum -= clock_intervals.popleft()

                    clock_intervals.append(clock_interval)
                    intervals_sum += clock_interval

                    if len(clock_intervals) == CLOCK_BUFFER_SIZE:
                        buf_clock_interval = intervals_sum / CLOCK_BUFFER_SIZE
                        bpm = 60 / (buf_clock_interval * 24)
                        print(f"BPM: {bpm:.2f}")
                last_clock_time = current_time
            clock_counter += 1
        if msg.type == 'quarter_frame':
            if not use_midi_beat_clock:
                mtc_type, value = parse_mtc(msg)
                last_mtc_timecode = update_mtc_timecode(mtc_type, value, clock_counter)
                hours = mtc_values[0] & 0x1F
                minutes = mtc_values[1]
                seconds = mtc_values[2]
                frames = mtc_values[3]
                current_total_frames = calculate_total_frames(hours, minutes, seconds, frames)
        elif msg.type == 'stop':  # Reset clock_counter and last_mtc_timecode on stop
            pass  # clock_counter = 0  # last_mtc_timecode = '00:00:00:00:00'  # mtc_values = [0, 0, 0, 0]
        elif msg.type == 'sysex' and msg.data[:5] == [0x7F, 0x7F, 0x01, 0x01, 0x00]:
            if not use_midi_beat_clock:
                mtc_values = [msg.data[5], msg.data[6], msg.data[7], msg.data[8]]
                for mtc_type, value in enumerate(mtc_values):
                    last_mtc_timecode = update_mtc_timecode(mtc_type * 2, value & 0x0F, clock_counter)
                    last_mtc_timecode = update_mtc_timecode(mtc_type * 2 + 1, value >> 4, clock_counter)
        elif msg.type == 'note_off':
            # handle note off message
            note = 0
        elif msg.type == 'note_on':
            note = msg.note
        elif msg.type == 'control_change' and msg.control == 1:
            # handle mod wheel message
            pass
        elif msg.type == 'pitchwheel':
            # handle pitch wheel message
            pass
        elif msg.type == 'control_change':
            # Handle All Notes Off message (CC number 123)
            if msg.control == 123:
                clock_counter = 0
                last_mtc_timecode = '00:00:00:00:00'
                mtc_values = [0, 0, 0, 0]
            # Handle Stop message (CC number 120)
            # elif msg.control == 51:
            #     clock_counter = 0
            #     last_mtc_timecode = '00:00:00:00:00'
            #     mtc_values = [0, 0, 0, 0]
            # Handle other control change messages
            elif msg.control == 1:
                # handle mod wheel message
                pass

    if last_mtc_timecode is not None or use_midi_beat_clock:
        hours = mtc_values[0] & 0x1F
        minutes = mtc_values[1]
        seconds = mtc_values[2]
        frames = mtc_values[3]
        current_total_frames = calculate_total_frames(hours, minutes, seconds, frames)

    if use_midi_beat_clock:
        current_total_frames = int(clock_counter / 24)

    return str(last_mtc_timecode), current_total_frames


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
                        bg_mask=4, buffer_size=2221):
    # Create a circular buffer (deque) for buff_png if it doesn't exist yet
    if not hasattr(display_png_filters, "buff_png"):
        display_png_filters.buff_png = deque(maxlen=buffer_size)

    effective_length = len(png_paths)

    # Define a function to search the buffer for a frame
    def find_frame_in_buffer(index, folder):
        for buffered_frame in display_png_filters.buff_png:
            if buffered_frame["main_folder"] == folder and (
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
        display_png_filters.buff_png.append({"index": index, "main_folder": folder, "frame": main_layer})

        if use_as_top_mask:
            display_png_filters.buff_png.append({"index": index, "main_folder": bg_mask, "frame": background_frame})

        return main_layer


def mtc_png_realtime(png_paths, input_port, listening_channels=1):
    while not stop_event.is_set():

        note_scaled = note % 12  # this scales the keys to one octave
        mtc_timecode_local, local_total_frames = process_midi_messages(input_port, listening_channels)
        index = calculate_index(local_total_frames, png_paths)
        time.sleep(.001)
        frame = display_png_filters(index, png_paths, note_scaled)
        if frame is not None:
            display_png_live(frame, mtc_timecode_local, local_total_frames, index)

        # Break the loop if the 'q' key is pressed
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed in (ord('q'), ord('f'), ord('c')):
            if key_pressed == ord('q'):
                stop_event.set()
            if key_pressed == ord('f'):
                print_memory_usage()
            if key_pressed == ord('c'):
                display_clock_time = not display_clock_time
    time.sleep(0.01)
    # Clean up
    input_port.close()
    cv2.destroyAllWindows()


def main():
    listening_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                          15]  # For example, listen to channels 1, 2, and 3
    midi_input_port = select_midi_input()  # Let user choose MIDI input port
    input_port = mido.open_input(midi_input_port)
    csv_source = select_csv_file()
    png_paths = parse_array_file(csv_source)
    mtc_png_realtime(png_paths, input_port, listening_channels)


if __name__ == "__main__":
    main()
