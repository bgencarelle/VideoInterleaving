import mido
import time
import math
from midi_control import select_midi_input


def calculate_jitter(expected_interval, actual_interval):
    return abs(actual_interval - expected_interval)


def calculate_percent_jitter(jitter, expected_interval):
    deviation = (jitter / expected_interval) * 100
    return deviation - 100


def calculate_snr_jitter(jitter, expected_interval):
    return 20 * math.log10(expected_interval / jitter)


def receive_mtc_smpte(midi_in, frame_rate):
    frame_durations = {
        0: 1 / 24,
        1: 1 / 25,
        2: 1 / 29.97,
        3: 1 / 30
    }
    expected_interval = frame_durations[frame_rate] / 4  # Divide by 4 since there are 4 messages per frame

    last_received_time = None
    jitter_sum = 0
    message_count = 0

    while True:
        msg = midi_in.receive(block=True)
        if msg.type == 'quarter_frame':
            current_time = time.time()

            if last_received_time is not None:
                actual_interval = current_time - last_received_time
                jitter = calculate_jitter(expected_interval, actual_interval)
                jitter_sum += jitter
                message_count += 1

                percent_jitter = calculate_percent_jitter(jitter, expected_interval)
                snr_jitter = calculate_snr_jitter(jitter, expected_interval)
                avg_jitter = jitter_sum / message_count

                # print(f"Jitter: {jitter:.6f} seconds, "
                #      f"Average jitter: {avg_jitter:.6f} seconds, "
                #      f"Percent jitter: {percent_jitter:.2f}%, "
                #      f"Jitter in SNR: {snr_jitter:.2f} dB")

            last_received_time = current_time


if __name__ == "__main__":
    midi_port = select_midi_input()
    midi_in = mido.open_input(midi_port)

    frame_rate = 3  # Choose your frame rate: 0 for 24 FPS, 1 for 25 FPS, 2 for 29.97 FPS, 3 for 30 FPS
    receive_mtc_smpte(midi_in, frame_rate)
