import csv
import os
import mido
import decimal
import math
from collections import deque


png_paths_len = 2221
frame_duration = 4.0
video_length = 19173
bpm_smoothing_window = 10


def get_midi_length(midi_file_path):
    midi_file = mido.MidiFile(midi_file_path)
    midi_length_seconds = midi_file.length
    frames_per_second = 30  # Assuming 30 FPS
    midi_length_frames = int(midi_length_seconds * frames_per_second)
    return midi_length_frames


def set_video_length(video_name, video_name_length):
    presets_folder = "presets"
    if not os.path.exists(presets_folder):
        os.makedirs(presets_folder)

    csv_file_path = os.path.join(presets_folder, "set_video_length.csv")

    with open(csv_file_path, mode="a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([video_name, video_name_length])


def get_video_length(video_number=0):
    with open("presets/set_video_length.csv", mode="r", newline='') as file:
        reader = csv.reader(file)
        video_lengths = list(reader)

        if len(video_lengths) == 1:
            return int(video_lengths[0][1])

        if 0 < video_number <= len(video_lengths):
            return int(video_lengths[video_number - 1][1])

        print("Please choose a video by name from the list:")
        for i, video in enumerate(video_lengths):
            print(f"{i+1}. {video[0]}")

        while True:
            selected_video = input("Enter the name of the video: ")
            for video in video_lengths:
                if video[0] == selected_video:
                    return int(video[1])
            print("Invalid video name. Please try again.")


def calculate_frame_duration(text_mode=True, setup_mode=False):
    global video_length, frame_duration
    if setup_mode:
        if text_mode:
            while True:
                try:
                    video_length = int(input("Please specify the length of the video in frames: "))
                    video_length = abs(video_length)
                    break
                except ValueError:
                    print("Invalid input. Please enter a positive integer.")
            video_name = input("Please enter the name of the video: ")
            set_video_length(video_name, video_length)
        else:
            midi_file_path = input("Please enter the MIDI file path: ")
            if not os.path.exists(midi_file_path):
                raise FileNotFoundError("MIDI file not found.")
            if not midi_file_path.lower().endswith('.mid'):
                raise ValueError("Invalid file type. Please provide a valid MIDI file.")
            video_length = get_midi_length(midi_file_path)
            video_name = input("Please enter the name of the video: ")
            set_video_length(video_name, video_length)
    else:
        video_length = get_video_length(video_number=0)
    frame_duration = video_length / png_paths_len
    print("Frame scaling factor for this video: ", frame_duration)
    return frame_duration


def select_csv_file():
    csv_dir = 'generatedPngLists'
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    if not csv_files:
        print(f"No .csv files found in directory {csv_dir}")
        return None

    if len(csv_files) == 1:
        print("Only one .csv file found, defaulting to:")
        selected_file = os.path.join(csv_dir, csv_files[0])
        print(selected_file)
        return selected_file

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


def get_image_names_from_csv(file_path):
    global png_paths_len
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row[1:] for row in csv_reader]
        png_paths_len = len(png_paths)
    return png_paths


def check_index_differences():
    previous_index = None
    positive_result_counts = {}
    negative_result_counts = {}
    video_length_mx = video_length * 20

    for i in range(video_length_mx):
        current_index, direction= calculate_index(i)

        # Count the unique results
        if direction > 0:
            if current_index not in positive_result_counts:
                positive_result_counts[current_index] = 0
            positive_result_counts[current_index] += 1
        else:
            if current_index not in negative_result_counts:
                negative_result_counts[current_index] = 0
            negative_result_counts[current_index] += 1

        # Check index differences
        if previous_index is not None and abs(current_index - previous_index) > 1:
            abs_diff_check = abs(current_index - previous_index)
            print(f"Index difference greater than 1 found between steps {i - 1} and {i}: {abs_diff_check}")

        previous_index = current_index

    print("Number of times each unique result appears:")
    sorted_positive_results = sorted(positive_result_counts.items())
    sorted_negative_results = sorted(negative_result_counts.items(), key=lambda x: abs(x[0]))

    for pos_result, pos_count in sorted_positive_results:
        for neg_result, neg_count in sorted_negative_results:
            if abs(pos_result) == abs(neg_result):
                diff_result = abs(pos_count - neg_count)
                if diff_result >= 1:
                    print(f"Result: {pos_result}, Count: {pos_count} --- Result: -{neg_result}, Count: {neg_count},"
                          f"diff: {diff_result}")
                break
    print(f"extremes 0: {positive_result_counts[0]}, pos {png_paths_len-1}: {positive_result_counts[png_paths_len-1]},"
          f"neg: {negative_result_counts[png_paths_len-1]}, ")


def calculate_index(frame_counter):
    scale_ref = 4.0

    frame_scale = scale_ref / frame_duration

    progress = (decimal.Decimal(frame_counter * frame_scale)) % (png_paths_len * 2)

    if progress < png_paths_len:
        index = int(progress.quantize(decimal.Decimal('1.000000'), rounding=decimal.ROUND_HALF_UP))
        direction = 1
    else:
        index = int((decimal.Decimal(png_paths_len * 2) - progress).quantize(decimal.Decimal('1.000000'), rounding=decimal.ROUND_HALF_UP))
        direction = -1

    # Ensure the index is within the valid range
    index = max(0, min(index, png_paths_len))

    # print(index * direction)
    return index, direction



def init_all():
    global frame_duration
    csv_source = select_csv_file()
    png_paths = get_image_names_from_csv(csv_source)
    # print(png_paths_len == len(png_paths), png_paths_len)
    frame_duration = calculate_frame_duration(True, False)

    return csv_source, png_paths


def main():
    init_all()
    # print(frame_dur)
    check_index_differences()


if __name__ == "__main__":
    main()
