import mido
from concurrent.futures import ThreadPoolExecutor
import threading
import sys


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

        print(f"Total frames: {self.total_frames}")


if __name__ == "__main__":
    midi_processor = MidiProcessor()
    midi_port = select_midi_input()

    with ThreadPoolExecutor() as executor:
        executor.submit(midi_processor.process_midi, midi_port)
        executor.submit(check_input_key, midi_processor)
