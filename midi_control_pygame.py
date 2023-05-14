import sys

import pygame
import pygame.midi
import calculators

mtc_received = False
recent_messages = []
total_frames = 0
frame_rate = 30
mtc_values = [0, 0, 0, 0]
index = 0
index_direction = 0


def calculate_total_frames():
    global total_frames

    hours = mtc_values[0] & 0x1F
    minutes = mtc_values[1]
    seconds = mtc_values[2]
    frames = mtc_values[3]
    total_frames = (hours * 60 * 60 * frame_rate) + \
                   (minutes * 60 * frame_rate) + (seconds * frame_rate) + frames
    total_frames = total_frames

    print(f"Total frames: {total_frames}")
    return total_frames


def is_complete_mtc_code():
    # Assuming all 8 messages have been received when the list has 8 elements
    return len(recent_messages) == 8


def calculate_time_code():
    hours = mtc_values[0] & 0x1F
    minutes = mtc_values[1]
    seconds = mtc_values[2]
    frames = mtc_values[3]

    frame_rate_code = (mtc_values[0] & 0x60) >> 5
    frame_rates = {0: 24, 1: 25, 2: 29.97, 3: 30}
    frame_rate = frame_rates[frame_rate_code]

    time_code = f'{hours:02}:{minutes:02}:{seconds:02}:{frames:02} @ {frame_rate} fps'
    print(f"Time code: {time_code}")



def process_mtc(msg):
    global mtc_received, recent_messages, total_frames, frame_rate, mtc_values, index, index_direction

    mtc_type, value = parse_mtc(msg)
    frame_updated = update_mtc_timecode(mtc_type, value)
    recent_messages.append(msg)

    if frame_updated:  # Call `calculate_time_code` only when the frame value has been updated
        calculate_time_code()

    if is_complete_mtc_code():
        mtc_received = True
        calculate_total_frames()
        index, direction = calculators.calculate_index(total_frames)
        mtc_received = False
        recent_messages.clear()


def update_mtc_timecode(mtc_type, value):
    global mtc_values

    frame_updated = False

    if mtc_type == 0:
        mtc_values[3] = (mtc_values[3] & 0xF0) | value
    elif mtc_type == 1:
        mtc_values[3] = (mtc_values[3] & 0x0F) | (value << 4)
        frame_updated = True
    elif mtc_type == 2:
        mtc_values[2] = (mtc_values[2] & 0xF0) | value
    elif mtc_type == 3:
        mtc_values[2] = (mtc_values[2] & 0x0F) | (value << 4)
    elif mtc_type == 4:
        mtc_values[1] = (mtc_values[1] & 0xF0) | value
    elif mtc_type == 5:
        mtc_values[1] = (mtc_values[1] & 0x0F) | (value << 4)
    elif mtc_type == 6:
        mtc_values[0] = (mtc_values[0] & 0xEF) | (value << 4)
    elif mtc_type == 7:
        mtc_values[0] = (mtc_values[0] & 0x1F) | (value << 4)

    return frame_updated


def parse_mtc(msg):
    status_byte = msg[0]
    mtc_type = (msg[1] & 0x70) >> 4
    value = msg[1] & 0x0F
    return mtc_type, value


def handle_note_off(msg):
    print(f"Note Off: Channel {msg[0] & 0x0F}, Note {msg[1]}, Velocity {msg[2]}")


def handle_note_on(msg):
    print(f"Note On: Channel {msg[0] & 0x0F}, Note {msg[1]}, Velocity {msg[2]}")


def handle_polyphonic_key_pressure(msg):
    pass
    # print(f"Polyphonic Key Pressure: Channel {msg[0] & 0x0F}, Note {msg[1]}, Pressure {msg[2]}")


def handle_control_change(msg):
    pass
    # print(f"Control Change: Channel {msg[0] & 0x0F}, Controller {msg[1]}, Value {msg[2]}")


def handle_program_change(msg):
    pass
    # print(f"Program Change: Channel {msg[0] & 0x0F}, Program {msg[1]}")


def handle_channel_pressure(msg):
    pass
    # print(f"Channel Pressure: Channel {msg[0] & 0x0F}, Pressure {msg[1]}")


def handle_pitch_bend(msg):
    value = (msg[2] << 7) | msg[1]
    # print(f"Pitch Bend: Channel {msg[0] & 0x0F}, Value {value}")


def handle_clock(msg):
    print("MIDI Clock")


def handle_quarter_frame(msg):
    print(f"MTC Quarter Frame: {msg[1]}")


def handle_sysex(msg):
    print(f"System Exclusive: {msg[1:]}")


def handle_midi_time_code_quarter_frame(msg):
    message_type = (msg[1] >> 4) & 0x07
    value = msg[1] & 0x0F
    print(f"MIDI Time Code Quarter Frame: Message Type {message_type}, Value {value}")


def handle_song_position_pointer(msg):
    value = (msg[2] << 7) | msg[1]
    print(f"Song Position Pointer: {value}")


def handle_song_select(msg):
    value = msg[1]
    print(f"Song Select: {value}")


def handle_tune_request(msg):
    print("Tune Request")


def handle_end_of_sysex(msg):
    print("End of System Exclusive")


def handle_system_realtime_message(msg):
    if msg[0] == 0xF8:
        handle_timing_clock()
    elif msg[0] == 0xFA:
        handle_start()
    elif msg[0] == 0xFB:
        handle_continue()
    elif msg[0] == 0xFC:
        handle_stop()
    elif msg[0] == 0xFE:
        handle_active_sensing()
    elif msg[0] == 0xFF:
        handle_reset()
    else:
        print(f"Unhandled system realtime message: {msg}")
        sys.exit(5)


def handle_timing_clock(msg=None):
    print("Timing Clock")


def handle_start(msg=None):
    print("Start")


def handle_continue(msg=None):
    print("Continue")


def handle_stop(msg=None):
    print("Stop")


def handle_active_sensing(msg=None):
    print("Active Sensing")


def handle_reset(msg=None):
    print("Reset")


def handle_system_message(msg):
    status_byte = msg[0]

    if status_byte == 0xF0 or status_byte == 0xF7:
        handle_sysex(msg)
    else:
        system_realtime_handlers = {
            0xF1: process_mtc,
            0xF2: handle_song_position_pointer,
            0xF3: handle_song_select,
            0xF6: handle_tune_request,
            0xF8: handle_timing_clock,
            0xFA: handle_start,
            0xFB: handle_continue,
            0xFC: handle_stop,
            0xFE: handle_active_sensing,
            0xFF: handle_reset
        }
        handler = system_realtime_handlers.get(status_byte)
        if handler:
            handler(msg)
        else:
            print(f"Unhandled System message: {hex(status_byte)}")


def process_message(msg):
    status_byte = msg[0]
    msg_type = status_byte & 0xF0

    message_handlers = {
        0x90: handle_note_on,
        0x80: handle_note_off,
        0xB0: handle_control_change,
        0xA0: handle_polyphonic_key_pressure,
        0xC0: handle_program_change,
        0xD0: handle_channel_pressure,
        0xE0: handle_pitch_bend,
    }

    if msg_type in message_handlers:
        handler = message_handlers[msg_type]
        handler(msg)
    else:
        handle_system_message(msg)


def midi_parser(callback, device_id=None):
    pygame.init()
    pygame.midi.init()

    if device_id is None:
        device_id = pygame.midi.get_default_input_id()

    if device_id == -1:
        print("No MIDI input devices found. Exiting.")
        return

    print(f"Using input device ID: {device_id}")

    midi_input = pygame.midi.Input(device_id)

    try:
        while True:
            if midi_input.poll():
                midi_events = pygame.midi.midis2events(midi_input.read(1), midi_input.device_id)
                for event in midi_events:
                    status_byte = event.status
                    data1 = event.data1
                    data2 = event.data2
                    callback([status_byte, data1, data2])
    except KeyboardInterrupt:
        print("Exiting...")

    finally:
        midi_input.close()
        pygame.midi.quit()
        pygame.quit()

def process_midi(device_id=None):
    midi_parser(process_message, device_id)

if __name__ == "__main__":
    midi_parser(process_message)
