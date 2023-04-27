import mido
from midi_mtc_png_display import select_midi_input
from concurrent.futures import ThreadPoolExecutor
import threading
import csv
import os
import sys

def write_total_frames_to_csv(total_frames, file_name='output.csv'):
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='', encoding='utf-8-sig') as csvfile:
            csv_writer = csv.writer(csvfile, dialect='excel')
            csv_writer.writerow(['Index', 'Total Frames', 'Difference'])

    # Read the current number of rows in the file to determine the index
    with open(file_name, 'r', newline='') as csvfile:
        current_rows = sum(1 for row in csv.reader(csvfile))
        index = current_rows - 1  # Subtract 1 for the header row
        difference = total_frames - 2 * index

    with open(file_name, 'a', newline='', encoding='utf-8-sig') as csvfile:
        csv_writer = csv.writer(csvfile, dialect='excel')
        csv_writer.writerow([index, total_frames, difference])


def check_input_key():
    while True:
        key = input()
        if key == 'q':
            sys.exit()


class MidiProcessor:
    def __init__(self):
        self.mtc_received = threading.Event()
        self.recent_messages = []
        self.total_frames = 0
        self.frame_rate = None
        self.mtc_values = [0, 0, 0, 0]

    def process_midi(self, port):
        with mido.open_input(port) as in_port:
            for msg in in_port:
                if msg.type == 'quarter_frame':
                    self.process_mtc(msg)
                elif msg.type in ('note_on', 'note_off', 'mod_wheel', 'program_change', 'clock'):
                    self.process_message(msg)

    def process_mtc(self, msg):
        mtc_type, value = self.parse_mtc(msg)
        self.update_mtc_timecode(mtc_type, value)
        self.recent_messages.append(msg)

        if self.is_complete_mtc_code():
            self.mtc_received.set()
            self.calculate_frame_rate()
            self.calculate_time_code()
            self.calculate_total_frames()
            self.mtc_received.clear()
            self.recent_messages.clear()

    def process_message(self, msg):
        pass
        # Process non-MTC messages as needed

    def is_complete_mtc_code(self):
        # Assuming all 8 messages have been received when the list has 8 elements
        return len(self.recent_messages) == 8

    def calculate_frame_rate(self):
        # Calculate the frame rate
        # Assuming 30 fps for now, you can update this as needed
        self.frame_rate = 30

    def calculate_time_code(self):
        hours = self.mtc_values[0] & 0x1F
        minutes = self.mtc_values[1]
        seconds = self.mtc_values[2]
        frames = self.mtc_values[3]
        time_code = f'{hours:02}:{minutes:02}:{seconds:02}:{frames:02}'
        print(f"Time code: {time_code}")

    def parse_mtc(self, msg):
        mtc_type = msg.frame_type
        value = msg.frame_value
        return mtc_type, value

    def update_mtc_timecode(self, mtc_type, value):
        if mtc_type == 0:
            self.mtc_values[3] = (self.mtc_values[3] & 0xF0) | value
        elif mtc_type == 1:
            self.mtc_values[3] = (self.mtc_values[3] & 0x0F) | (value << 4)
        elif mtc_type == 2:
            self.mtc_values[2] = (self.mtc_values[2] & 0xF0) | value
        elif mtc_type == 3:
            self.mtc_values[2] = (self.mtc_values[2] & 0x0F) | (value << 4)
        elif mtc_type == 4:
            self.mtc_values[1] = (self.mtc_values[1] & 0xF0) | value
        elif mtc_type == 5:
            self.mtc_values[1] = (self.mtc_values[1] & 0x0F) | (value << 4)
        elif mtc_type == 6:
            self.mtc_values[0] = (self.mtc_values[0] & 0xF0) | value
        elif mtc_type == 7:
            self.mtc_values[0] = (self.mtc_values[0] & 0x0F) | (value << 4)

    def calculate_total_frames(self):
        hours = self.mtc_values[0] & 0x1F
        minutes = self.mtc_values[1]
        seconds = self.mtc_values[2]
        frames = self.mtc_values[3]
        total_frames = (hours * 60 * 60 * self.frame_rate) + \
                       (minutes * 60 * self.frame_rate) + (seconds * self.frame_rate) + frames
        self.total_frames = total_frames
        write_total_frames_to_csv(self.total_frames)

        # print(f"Total frames: {self.total_frames}")


if __name__ == "__main__":
    midi_input = select_midi_input()
    midi_processor = MidiProcessor()

    with ThreadPoolExecutor() as executor:
        midi_port = midi_input
        executor.submit(midi_processor.process_midi, midi_port)
        executor.submit(check_input_key, midi_processor)