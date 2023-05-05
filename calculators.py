import csv
import os
import mido

png_paths_len = 1
frame_duration = 1.0
video_length = 1


def get_midi_length(midi_file_path):
    midi_file = mido.MidiFile(midi_file_path)
    midi_length_seconds = midi_file.length
    frames_per_second = 30  # Assuming 30 FPS
    midi_length_frames = int(midi_length_seconds * frames_per_second)
    return midi_length_frames


def calculate_frame_duration(text_mode=True):
    global png_paths_len, frame_duration, video_length
    if text_mode:
        video_length = int(input("Please specify the length of the video in frames: "))
        if video_length <= 30:
            raise ValueError("Length should be greater than 30.")
    else:
        midi_file_path = input("Please enter the MIDI file path: ")
        if not os.path.exists(midi_file_path):
            raise FileNotFoundError("MIDI file not found.")
        if not midi_file_path.lower().endswith('.mid'):
            raise ValueError("Invalid file type. Please provide a valid MIDI file.")
        video_length = get_midi_length(midi_file_path)

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
    result_counts = {}
    video_length_mx = video_length

    for i in range(video_length_mx):
        current_index, _ = calculate_index(i*2)

        # Count the unique results
        if current_index not in result_counts:
            result_counts[current_index] = 0
        result_counts[current_index] += 1

        # Check index differences
        if previous_index is not None and abs(current_index - previous_index) > 1:
            print(f"Index difference greater than 1 found between steps {i - 1} and {i}")

        previous_index = current_index

    print("Number of times each unique result appears:")
    for result, count in result_counts.items():
        print(f"Result: {result}, Count: {count}")


def calculate_index(estimate_frame_counter, index_mult=8.0):
    global frame_duration, png_paths_len
    if index_mult >= frame_duration*.5:
        index_mult = frame_duration*.5
    frame_scale = 1 / (frame_duration / index_mult)
    progress = (estimate_frame_counter * frame_scale) % (png_paths_len * 2)

    if int(progress) <= png_paths_len:
        index = int(progress)
        direction = 1
    else:
        index = int(png_paths_len * 2 - progress)
        direction = -1

    # Ensure the index is within the valid range
    index = max(0, min(index, png_paths_len - 1))
    print(index, direction)

    return index, direction


def init_all():
    csv_source = select_csv_file()
    get_image_names_from_csv(csv_source)
    # print(png_paths_len == len(png_paths), png_paths_len)
    calculate_frame_duration()


def main():
    init_all()
    # print(frame_dur)
    check_index_differences()


if __name__ == "__main__":
    main()
