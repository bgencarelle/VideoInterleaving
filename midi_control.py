import time
from collections import deque

import mido

import calculators


CLOCK_BUFFER_SIZE = 200
TIMEOUT_SECONDS = 1  # Set the timeout value as needed

clock_counter_sum = 0
clock_mode = 1  # 0 is mtc, 1 is midi_clock
frame_duration = 1
video_len = 1
source_frame_total = 1
mtc_received = False
recent_messages = []
mtc_values = [0, 0, 0, 0]
input_port = None
index = 0
frame_rate = 30
direction = 1
clock_index = 0
clock_index_direction = 1
last_received_mtc_time = 0  # Initialize last received MTC message time
total_frames = 0
frame_type_count = {i: 0 for i in range(8)}  # Initialize frame type count
new_mtc_code_started = [False]
bpm = 120


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


def process_midi(mode=clock_mode):
    global clock_mode
    clock_mode = mode
    in_port = input_port
    for msg in in_port.iter_pending():
        if msg.type == 'quarter_frame':
            process_mtc(msg)
        elif msg.type in ('note_on', 'note_off', 'mod_wheel', 'program_change', 'clock'):
            process_message(msg)
    return


def process_mtc(msg):
    global mtc_received, frame_rate, mtc_values, index, direction, total_frames, clock_counter_sum
    mtc_type, value = parse_mtc(msg)
    update_mtc_timecode(mtc_type, value)

    if mtc_type == 7:  # Recalculate timecode once FRAMES LSB quarter-frame is received
        old_total_frames = total_frames
        old_total_clocks = clock_counter_sum
        mtc_received = True
        calculate_time_code()
        total_frames = calculate_total_frames()
        frames_by_2 = total_frames // 2
        clock_counter_sum = clock_counter()
        clock_diff = (clock_counter_sum - old_total_clocks) * 24
        total_frame_diff = abs(total_frames - old_total_frames)
        #print("clock_diff", clock_diff)
        if total_frame_diff > 2:
            print(f'difference: {total_frame_diff},at frame: {total_frames}')
        # print(bpm)
        index, direction = calculators.calculate_index(frames_by_2 + clock_diff)
        # print(bpm, "INDEX:", index*direction, "FRAMES:", total_frames)

        mtc_received = False


def update_mtc_timecode(mtc_type, value):
    global mtc_values

    # Update the buffer with the received quarter-frame value
    mtc_values[mtc_type] = value


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


def handle_note_on(msg):
    print(f"Note on: {msg.note}, velocity: {msg.velocity}, channel: {msg.channel}")


def handle_note_off(msg):
    print(f"Note off: {msg.note}, velocity: {msg.velocity}, channel: {msg.channel}")


def handle_mod_wheel(msg):
    print(f"Modulation wheel: value: {msg.value}, channel: {msg.channel}")


def handle_program_change(msg):
    print(f"Program change: program: {msg.program}, channel: {msg.channel}")


def handle_clock(msg):
    global clock_index, clock_index_direction, bpm

    if not hasattr(handle_clock, "clock_intervals"):
        handle_clock.clock_intervals = deque(maxlen=CLOCK_BUFFER_SIZE)
        handle_clock.intervals_sum = 0
        handle_clock.last_clock_time = None

    current_time = time.time()

    # Check if the timeout has been exceeded
    if handle_clock.last_clock_time is not None and current_time - handle_clock.last_clock_time > TIMEOUT_SECONDS:
        print("Clock messages stopped. No BPM update.")
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
                # print(f"BPM: {bpm:.2f}")

    handle_clock.last_clock_time = current_time
    # Your original code
    clock_counter(1)
    if clock_mode == 1:
        clock_index, clock_index_direction = calculators.calculate_index(clock_counter(), False)


def clock_counter(amount=None):
    if not hasattr(clock_counter, "counter"):
        clock_counter.counter = 0

    if amount is not None:
        clock_counter.counter += amount
    # print(clock_counter.counter)
    return clock_counter.counter


def process_message(msg):
    handler = message_handlers.get(msg.type)
    if handler:
        handler(msg)
    else:
        print(f"Unhandled message type: {msg.type}")


message_handlers = {
    'note_on': handle_note_on,
    'note_off': handle_note_off,
    'mod_wheel': handle_mod_wheel,
    'program_change': handle_program_change,
    'clock': handle_clock,
}


def midi_control_stuff_main():
    global input_port
    midi_port = select_midi_input()
    input_port = mido.open_input(midi_port)


def main():
    midi_control_stuff_main()
    calculators.init_all()
    while True:
        process_midi()


if __name__ == "__main__":
    main()
