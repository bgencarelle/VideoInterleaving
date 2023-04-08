import mido
import cv2
import csv
import numpy as np
import os
from threading import Event
import random

highest_timecode_seen = 0
clock_counter = 0
stop_event = Event()
mtc_values = [0, 0, 0, 0]

mtc_img = np.zeros((150, 600), dtype=np.uint8)
last_mtc_timecode = '00:00:00:00:00'


def parse_array_file(file_path):
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row[1:] for row in csv_reader]

    return png_paths


def parse_mtc(msg):
    mtc_type = msg.frame_type
    value = msg.frame_value
    return mtc_type, value


def update_mtc_timecode(mtc_type, value, clock_counter):
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
    subframes = int((clock_counter / 24) * 100)  # Assuming 24 PPQN (Pulses Per Quarter Note)

    return f'{hours:02}:{minutes:02}:{seconds:02}:{frames:02}:{subframes:02}'


def calculate_total_frames(hours, minutes, seconds, frames):
    total_frames = (hours * 60 * 60 * 30) + (minutes * 60 * 30) + (seconds * 30) + frames
    return total_frames


def process_midi_messages(input_port):
    global clock_counter, mtc_values, frame_counter, last_mtc_timecode
    current_total_frames = 0

    for msg in input_port.iter_pending():
        if msg.type == 'quarter_frame':
            mtc_type, value = parse_mtc(msg)
            last_mtc_timecode = update_mtc_timecode(mtc_type, value, clock_counter)
        elif msg.type == 'clock':
            clock_counter += 1
        elif msg.type == 'sysex' and msg.data[:5] == [0x7F, 0x7F, 0x01, 0x01, 0x00]:
            clock_counter = 0
            mtc_values = [msg.data[5], msg.data[6], msg.data[7], msg.data[8]]
            for mtc_type, value in enumerate(mtc_values):
                last_mtc_timecode = update_mtc_timecode(mtc_type * 2, value & 0x0F, clock_counter)
                last_mtc_timecode = update_mtc_timecode(mtc_type * 2 + 1, value >> 4, clock_counter)
        elif msg.type == 'note_off':
            # handle note off message
            pass
        elif msg.type == 'note_on':
            # handle note on message
            pass
        elif msg.type == 'control_change' and msg.control == 1:
            # handle mod wheel message
            pass
        elif msg.type == 'pitchwheel':
            # handle pitch wheel message
            pass

    if last_mtc_timecode is not None:
        hours = mtc_values[0] & 0x1F
        minutes = mtc_values[1]
        seconds = mtc_values[2]
        frames = mtc_values[3]
        current_total_frames = calculate_total_frames(hours, minutes, seconds, frames)

    return last_mtc_timecode, current_total_frames


def calculate_index(estimate_frame_counter, png_paths, index_mult=2.0, frame_duration=8.6326):
    effective_length = len(png_paths)
    index = int(estimate_frame_counter / (frame_duration / index_mult)) % (effective_length * 2)
    if index >= effective_length:
        index = effective_length - (index - effective_length) - 1
    return index


def display_png_live(frame, mtc_timecode, estimate_frame_counter, index):
    # Display timecode and other information
    cv2.putText(frame, mtc_timecode, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    cv2.putText(frame, f'Estimate Frame Counter: {estimate_frame_counter}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'index: {index}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(f'MTC Timecode - PID: {os.getpid()}', frame)


def display_png_filters(index, png_paths, folder=0, openCV_filters=None):
    if 0 <= index < len(png_paths):
        png_file = png_paths[index][folder]
        frame = cv2.imread(png_file, cv2.IMREAD_UNCHANGED)

        if frame is not None:
            # Resize the image
            frame = cv2.resize(frame, (900, 1200))

            # Check for transparency
            # if frame.shape[2] == 4:
            #     alpha_channel = frame[:, :, 3]
            #     _, mask = cv2.threshold(alpha_channel, 128, 255, cv2.THRESH_BINARY)
            #     frame = cv2.bitwise_and(frame, frame, mask=mask)
            #     frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Apply OpenCV filters (dummy code for now)
            if openCV_filters:
                pass

            return frame


def mtc_png_realtime():
    global clock_counter
    input_port = mido.open_input('IAC Driver Bus 1')
    csv_source = select_csv_file()
    png_paths = parse_array_file(csv_source)
    midi_note = 64
    note_scaled = midi_note % 12 # this scales the keys to one octave
    counter = 0
    old_timecode = 0
    while not stop_event.is_set():
        mtc_timecode_local, estimate_frame_counter_local = process_midi_messages(input_port)
        index = calculate_index(estimate_frame_counter_local, png_paths)
        # In the mtc_png_realtime() function, modify the display_png_live line:
        frame = display_png_filters(index, png_paths, counter)
        if frame is not None:
            display_png_live(frame, mtc_timecode_local, estimate_frame_counter_local, index)
        old_timecode = mtc_timecode_local

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

    # Clean up
    input_port.close()
    cv2.destroyAllWindows()


def select_csv_file():
    csv_dir = 'generatedPngLists'
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No .csv files found in directory {csv_dir}")
        return None

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


if __name__ == "__main__":
    mtc_png_realtime()