import mido
import calculators


frame_duration = 1
video_len = 1
source_frame_total = 1
mtc_received = False
recent_messages = []
total_frames = 0
frame_rate = 30
mtc_values = [0, 0, 0, 0]
midi_port = None
index = 0
direction = 1
clock_index = 0
clock_index_direction = 1


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


def process_midi():
    with mido.open_input(midi_port) as in_port:
        for msg in in_port:
            if msg.type == 'quarter_frame':
                process_mtc(msg)
            elif msg.type in ('note_on', 'note_off', 'mod_wheel', 'program_change', 'clock'):
                process_message(msg)


def process_mtc(msg):
    global mtc_received, recent_messages, total_frames, frame_rate, mtc_values, index, direction

    mtc_type, value = parse_mtc(msg)
    update_mtc_timecode(mtc_type, value)
    recent_messages.append(msg)

    if is_complete_mtc_code():
        mtc_received = True
        calculate_frame_rate()
        calculate_time_code()
        calculate_total_frames()
        index, direction = calculators.calculate_index(total_frames)
        mtc_received = False
        recent_messages.clear()


def is_complete_mtc_code():
    # Assuming all 8 messages have been received when the list has 8 elements
    return len(recent_messages) == 8


def calculate_frame_rate():
    global frame_rate
    # Calculate the frame rate
    # Assuming 30 fps for now, you can update this as needed
    frame_rate = 30


def calculate_time_code():
    hours = mtc_values[0] & 0x1F
    minutes = mtc_values[1]
    seconds = mtc_values[2]
    frames = mtc_values[3]
    time_code = f'{hours:02}:{minutes:02}:{seconds:02}:{frames:02}'
    #print(f"Time code: {time_code}")


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
    global total_frames, frame_rate, mtc_values

    hours = mtc_values[0] & 0x1F
    minutes = mtc_values[1]
    seconds = mtc_values[2]
    frames = mtc_values[3]
    total_frames = (hours * 60 * 60 * frame_rate) + \
                   (minutes * 60 * frame_rate) + (seconds * frame_rate) + frames
    total_frames = total_frames

    # print(f"Total frames: {total_frames}")
    return total_frames


def handle_note_on(msg):
    print(f"Note on: {msg.note}, velocity: {msg.velocity}, channel: {msg.channel}")


def handle_note_off(msg):
    print(f"Note off: {msg.note}, velocity: {msg.velocity}, channel: {msg.channel}")


def handle_mod_wheel(msg):
    print(f"Modulation wheel: value: {msg.value}, channel: {msg.channel}")


def handle_program_change(msg):
    print(f"Program change: program: {msg.program}, channel: {msg.channel}")


def handle_clock(msg):
    global clock_index_direction
    global clock_index
    clock_counter(1)
    clock_index, clock_index_direction = calculators.calculate_index(clock_counter())
    # print("Clock message received")


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
    global midi_port
    midi_port = select_midi_input()

def main():
    calculators.init_all()
    process_midi()
if __name__ == "__main__":
    main()