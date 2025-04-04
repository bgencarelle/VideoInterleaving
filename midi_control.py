# this handles all the midi stuff
import time
from collections import deque

import mido
import mido.backends.rtmidi
import mido.backends.backend
import mido.backends.pygame

import calculators

from settings import (MTC_CLOCK, MIDI_CLOCK, MIXED_CLOCK,
                    CLOCK_BUFFER_SIZE, CLOCK_MODE, TIMEOUT_SECONDS)


# globals, sorry not sorry
png_paths_len = 0
clock_counter_sum = 0
frame_duration = 1
video_len = 1
source_frame_total = 1
clock_frame_ratio = 1
mtc_received = False
recent_messages = []
mtc_values = [0, 0, 0, 0]
input_port = None
index = 0
frame_rate = 30
index_direction = 1
last_received_mtc_time = 0  # Initialize last received MTC message time
total_frames = 0
frame_type_count = {i: 0 for i in range(8)}  # Initialize frame type count
new_mtc_code_started = [False]
bpm = 120

midi_data_dictionary = {
    'Note_On': (0, 127, 0),
    'Note_Off': (None, None, None),
    'Modulation': (0, 0),
    'Index_and_Direction': (0, 1),
    'BPM': 120,
    # 'Stop': False,
    # 'Start': False,
    # 'Pause': False,
    # 'Reset': False
}


def select_midi_input(midi_in_select=False):
    available_ports = mido.get_input_names()
    # Check if there are any available ports
    if len(available_ports) == 0:
        print("No MIDI input ports available.")
        return None

    # Check if there is only one port or all ports have the same name
    if len(set(available_ports)) == 1 or not midi_in_select:
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


def process_midi(mode=CLOCK_MODE):
    global clock_mode
    clock_mode = mode
    in_port = input_port
    for msg in in_port.iter_pending():
        if msg.type == 'quarter_frame' and clock_mode != 1:
            process_mtc(msg)
        elif msg.type in (
                'control_change',
                'note_on',
                'note_off',
                'mod_wheel',
                'program_change',
                'clock',
                'stop',
                'start',
                'continue',
                'songpos',
                'reset',
                'active_sensing',
                'pitchwheel'
        ):
            process_message(msg)
    return


def process_mtc(msg):
    global mtc_received, frame_rate, mtc_values, index, index_direction, total_frames, clock_counter_sum, \
        clock_frame_ratio
    mtc_type, value = parse_mtc(msg)
    update_mtc_timecode(mtc_type, value)

    if mtc_type == 7:  # Recalculate timecode once FRAMES LSB quarter-frame is received
        mtc_received = True
        old_total_frames = total_frames
        calculate_time_code()
        total_frames = calculate_total_frames()
        if total_frames < 4:
            clock_counter(0)
            clock_frame_ratio = 0
        total_frame_diff = abs(total_frames-old_total_frames)
        if total_frame_diff > 8:
            print(f'difference: {total_frame_diff},at frame: {total_frames}')
            clock_counter(0)
            clock_counter(clock_frame_ratio * total_frames)
        if clock_mode == MTC_CLOCK:
            index, index_direction = calculators.calculate_index(total_frames)
            midi_data_dictionary['Index_and_Direction'] = (index, index_direction)
        clock_counter_sum = clock_counter()
        if total_frames >= 4:
            clock_frame_ratio = clock_counter_sum / total_frames
        # print("clock counts since 0: ", clock_counter_sum, "clock_frame_ratio * frames: ", ratio_frames)
        # print("bpm: ", bpm, "INDEX: ", index*index_direction)
        mtc_received = False


def calculate_time_code():
    global frame_rate
    hours = mtc_values[0] & 0x1F
    minutes = mtc_values[1]
    seconds = mtc_values[2]
    frames = mtc_values[3]

    frame_rate_code = (mtc_values[0] & 0x60) >> 5
    frame_rates = {0: 24, 1: 25, 2: 29.97, 3: 30}
    frame_rate = frame_rates[frame_rate_code]

    time_code = f'{hours:02}:{minutes:02}:{seconds:02}:{frames:02} @ {frame_rate} fps'
    return time_code
    # print(f"Time code: {time_code}")


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


def calculate_total_frames():
    hours = mtc_values[0] & 0x1F
    minutes = mtc_values[1]
    seconds = mtc_values[2]
    frames = mtc_values[3]
    calc_frames = (hours * 60 * 60 * frame_rate) + \
                  (minutes * 60 * frame_rate) + (seconds * frame_rate) + frames

    # print(f"Total frames: {total_frames}")
    return calc_frames


def handle_stop(msg):
    global total_frames
    clock_counter(0)
    total_frames = 0
    print(f"STOP")


def handle_note_on(msg):
    midi_data_dictionary['Note_On'] = (msg.note, msg.velocity, msg.channel)
    # print(f"Note on: {msg.note}, velocity: {msg.velocity}, channel: {msg.channel}")


def handle_note_off(msg):
    midi_data_dictionary['Note_Off'] = (msg.note, msg.velocity, msg.channel)
    # print(f"Note off: {msg.note}, velocity: {msg.velocity}, channel: {msg.channel}")


def handle_mod_wheel(msg):
    midi_data_dictionary['Modulation'] = (msg.value, msg.channel)
    # print(f"Modulation wheel: value: {msg.value}, channel: {msg.channel}")


def handle_program_change(msg):
    pass
    # print(f"Program change: program: {msg.program}, channel: {msg.channel}")


def handle_clock(msg):
    global index, index_direction, bpm, total_frames

    if not hasattr(handle_clock, "clock_intervals"):
        handle_clock.clock_intervals = deque(maxlen=CLOCK_BUFFER_SIZE)
        handle_clock.intervals_sum = 0
        handle_clock.last_clock_time = None
        handle_clock.last_total_frames = None
        handle_clock.last_total_frames_time = None

    current_time = time.time()

    # Check if the timeout has been exceeded
    if handle_clock.last_clock_time is not None and current_time - handle_clock.last_clock_time > TIMEOUT_SECONDS:
        clock_counter(0)
        total_frames = 0
        print("Clock messages stopped.")
    else:
        if handle_clock.last_clock_time is not None:
            clock_interval = current_time - handle_clock.last_clock_time

            if len(handle_clock.clock_intervals) == CLOCK_BUFFER_SIZE:
                # Remove the oldest interval from the sum
                handle_clock.intervals_sum -= handle_clock.clock_intervals.popleft()

            handle_clock.clock_intervals.append(clock_interval)
            handle_clock.intervals_sum += clock_interval

            if len(handle_clock.clock_intervals) == CLOCK_BUFFER_SIZE:
                avg_clock_interval = handle_clock.intervals_sum / CLOCK_BUFFER_SIZE
                bpm = 60 / (avg_clock_interval * 24)
                midi_data_dictionary['BPM'] = bpm
                # print(f"BPM: {bpm:.2f}")

    handle_clock.last_clock_time = current_time

    # Check if total_frames has not changed in 200ms
    if clock_mode == MTC_CLOCK:
        if handle_clock.last_total_frames is not None and handle_clock.last_total_frames == total_frames\
                and current_time - handle_clock.last_total_frames_time > 0.2:
            clock_counter(0)
            print("Total frames haven't changed in 200ms. Clock reset to zero.")
        else:
            handle_clock.last_total_frames = total_frames
            handle_clock.last_total_frames_time = current_time

    # Your original code
    clock_counter(1)
    if clock_mode == MIDI_CLOCK:
        index, index_direction = calculators.calculate_index(clock_counter())
        midi_data_dictionary['Index_and_Direction'] = (index, index_direction)
    elif clock_mode == MIXED_CLOCK:
        index, index_direction = calculators.calculate_index(clock_counter() * clock_frame_ratio)
        midi_data_dictionary['Index_and_Direction'] = (index, index_direction)
    # print(f'clock counter: {clock_counter()}, Index: {index * index_direction}')


def handle_start(msg):
    global total_frames
    clock_counter(0)
    total_frames = 0
    print("Start message received.")
    # Add your handling code for the Start message here


def handle_continue(msg):
    global total_frames
    clock_counter(0)
    total_frames = 0
    print("Continue message received.", total_frames)
    # Add your handling code for the Continue message here


def handle_song_position_pointer(msg):
    global total_frames
    clock_counter(0)
    total_frames = 0
    print(f"Song Position Pointer message received. Value: {msg.pos}")
    # Add your handling code for the Song Position Pointer message here


def handle_system_reset(msg):
    global total_frames
    clock_counter(0)
    total_frames = 0
    print("System Reset message received.")
    # Add your handling code for the System Reset message here


def handle_active_sensing(msg):
    # global total_frames
    # clock_counter(0)
    # total_frames = 0
    print("ACTIVEEEEEE")
    # Add your handling code for the System Reset message here


def handle_pitchwheel(msg):
    pass
    # print("pitchwheel fun:", msg)


def clock_counter(amount=None):
    if not hasattr(clock_counter, "counter"):
        clock_counter.counter = 0

    if amount is not None:
        if amount == 0 or total_frames <= 1 and clock_mode != MIDI_CLOCK:
            clock_counter.counter = 0
        else:
            clock_counter.counter += int(amount)
    # print("clock_counter.counter says: ", clock_counter.counter)
    return clock_counter.counter


def process_message(msg):
    # print(msg.type)
    if msg.type == 'control_change':
        handle_control_change(msg)
    else:
        handler = message_handlers.get(msg.type)
        if handler:
            handler(msg)
        else:
            print(f"Unhandled message type: {msg.type}")


message_handlers = {
    'note_on': handle_note_on,
    'note_off': handle_note_off,
    'program_change': handle_program_change,
    'pitchwheel': handle_pitchwheel,
    'clock': handle_clock,
    'stop': handle_stop,
    'start': handle_start,
    'continue': handle_continue,
    'songpos': handle_song_position_pointer,
    'system_reset': handle_system_reset,
    'active_sensing': handle_active_sensing,
}


def handle_control_change(msg):
    if msg.control == 1:
        handle_mod_wheel(msg)
    else:
        pass  # print(f"Unhandled control change: {msg.control}")


def handle_mod_wheel(msg):
    midi_data_dictionary['Modulation'] = (msg.value, msg.channel)
    print(f"Modulation wheel: value: {msg.value}, channel: {msg.channel}")


def midi_control_stuff_main():
    global input_port, png_paths_len
    midi_port = select_midi_input()
    input_port = mido.open_input(midi_port)


def main():
    midi_control_stuff_main()
    while True:
        process_midi()


if __name__ == "__main__":
    main()
