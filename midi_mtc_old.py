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


def parse_mtc(msg):
    mtc_type = msg.frame_type
    value = msg.frame_value
    return mtc_type, value


def update_mtc_timecode(mtc_type, value):
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
    frame_count = calculate_total_frames(hours, minutes, seconds, frames)

    return total, frame_count


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


def calculate_index(frame_count, total_frames, index_mult=4.0, frame_duration=8.6326, smoothing_factor=0.1):
    effective_length = total_frames
    progress = (estimate_frame_count / (frame_duration / index_mult)) % (effective_length * 2)

    if int(progress) <= effective_length:
        index = int(progress)
    else:
        index = int(effective_length * 2 - progress)

    # Apply the smoothing factor to the index calculation
    smoothed_index = int(index * (1 - smoothing_factor) + smoothing_factor * (index + 1))

    # Ensure the index is within the valid range
    index = max(0, min(smoothed_index, effective_length - 1))

    return index




def mtc_png_realtime(input_port, listening_channels=1):
    mtc_timecode_local, local_total_frames = process_midi_messages(input_port, listening_channels)
    frame_count =
    index = calculate_index(frame_count, local_total_frames)
    input_port.close()


def main():
    listening_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                          15]  # For example, listen to channels 1, 2, and 3
    midi_input_port = select_midi_input()  # Let user choose MIDI input port
    input_port = mido.open_input(midi_input_port)
    mtc_png_realtime(input_port, listening_channels)


if __name__ == "__main__":
    main()
