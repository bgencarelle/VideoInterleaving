import mido
import time
from midi_control import select_midi_input
def generate_mtc_smpte_quarter_frame_messages(frame_rate, hours, minutes, seconds, frames):
    quarter_frame_data = [
        (frames & 0x0F),
        (frames >> 4) | 0x10,
        (seconds & 0x0F) | 0x20,
        (seconds >> 4) | 0x30,
        (minutes & 0x0F) | 0x40,
        (minutes >> 4) | 0x50,
        (hours & 0x0F) | (frame_rate << 1) | 0x60,
        (hours >> 4) | 0x70,
    ]

    quarter_frame_messages = [mido.Message('quarter_frame', frame_type=i, frame_value=value & 0x0F) for i, value in
                              enumerate(quarter_frame_data)]
    return quarter_frame_messages


def send_mtc_smpte(midipipe, frame_rate, hours, minutes, seconds, frames):
    mtc_messages = generate_mtc_smpte_quarter_frame_messages(frame_rate, hours, minutes, seconds, frames)

    for msg in mtc_messages:
        midipipe.send(msg)




if __name__ == "__main__":
    # MIDI pipe configuration
    midi_port = select_midi_input()
    midi_port_name = midi_port  # Replace with the name of your MIDI output port
    midi_out = mido.open_output(midi_port_name)

    # SMPTE/MTC configuration
    frame_rate = 3  # Choose your frame rate: 0 for 24 FPS, 1 for 25 FPS, 2 for 29.97 FPS, 3 for 30 FPS
    hours, minutes, seconds, frames = 0, 0, 0, 0

    while True:
        send_mtc_smpte(midi_out, frame_rate, hours, minutes, seconds, frames)
        frames += 1

        # Time increment based on chosen frame rate
        if frame_rate in (0, 3):
            time.sleep(1 / 30)
        elif frame_rate == 1:
            time.sleep(1 / 25)
        elif frame_rate == 2:
            time.sleep(1 / 29.97)

        # Update timecode
        if frames >= (30 if frame_rate == 3 else 25 if frame_rate == 1 else 29 if frame_rate == 2 else 24):
            frames = 0
            seconds += 1
            if seconds >= 60:
                seconds = 0
                minutes += 1
                if minutes >= 60:
                    minutes = 0
                    hours += 1
                    if hours >= 24:
                        hours = 0