import csv
import os
import mido
import decimal
import make_file_lists

png_paths_len = 2221
frame_duration = 4.0
video_length = 9173
bpm_smoothing_window = 10

# Constant for Free Clock mode
FREE_CLOCK = 255


def get_midi_length(midi_file_path):
    midi_file = mido.MidiFile(midi_file_path)
    midi_length_seconds = midi_file.length
    frames_per_second = 30  # Assuming 30 FPS
    return int(midi_length_seconds * frames_per_second)


def set_video_length(video_name, video_name_length):
    presets_folder = "presets"
    if not os.path.exists(presets_folder):
        os.makedirs(presets_folder)
    csv_file_path = os.path.join(presets_folder, "set_video_length.csv")
    with open(csv_file_path, mode="a", newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow([video_name, video_name_length])


def get_video_length(video_number=0):
    presets_folder = "presets"
    csv_file_path = os.path.join(presets_folder, "set_video_length.csv")
    # If the presets data is missing, run the setup.
    if not os.path.exists(csv_file_path):
        print("No video length data found. Running setup.")
        calculate_frame_duration()
    with open(csv_file_path, mode="r", newline='') as file:
        reader = csv.reader(file)
        video_lengths = list(reader)
        if len(video_lengths) == 1:
            return int(video_lengths[0][1])
        if 0 < video_number <= len(video_lengths):
            return int(video_lengths[video_number - 1][1])
        # If more than one video exists, prompt for a choice.
        print("Multiple videos found. Please choose one:")
        for video in video_lengths:
            print(video[0])
        while True:
            selected_video = input("Enter the name of the video: ").strip()
            for video in video_lengths:
                if video[0] == selected_video:
                    return int(video[1])
            print("Invalid video name. Please try again.")


def calculate_frame_duration():
    global video_length, frame_duration
    presets_folder = "presets"
    csv_file_path = os.path.join(presets_folder, "set_video_length.csv")
    # If no preset exists, interactively ask for video length.
    if not os.path.exists(csv_file_path):
        print("No video length preset found. Please set up video length.")
        mode = input("Enter 'm' for manual entry or 'd' for MIDI-derived length: ").strip().lower()
        if mode == 'm':
            while True:
                try:
                    video_length = int(input("Enter the video length in frames: "))
                    video_length = abs(video_length)
                    break
                except ValueError:
                    print("Invalid input. Please enter a positive integer.")
        elif mode == 'd':
            while True:
                midi_file_path = input("Enter the MIDI file path: ").strip()
                if not os.path.exists(midi_file_path):
                    print("MIDI file not found. Try again.")
                elif not midi_file_path.lower().endswith('.mid'):
                    print("Invalid file type. Please provide a MIDI file.")
                else:
                    try:
                        video_length = get_midi_length(midi_file_path)
                        break
                    except Exception as e:
                        print(f"Error: {e}")
        else:
            print("Invalid mode selected; defaulting to manual entry.")
            while True:
                try:
                    video_length = int(input("Enter the video length in frames: "))
                    video_length = abs(video_length)
                    break
                except ValueError:
                    print("Invalid input. Please enter a positive integer.")
        video_name = input("Enter the name of the video: ").strip()
        set_video_length(video_name, video_length)
    else:
        # If the preset exists, load its value.
        video_length = get_video_length()
    frame_duration = video_length / png_paths_len
    print("Frame scaling factor for this video:", frame_duration)
    return frame_duration


def select_img_list_files():
    csv_dir = 'generated_img_lists'
    if not os.path.exists(csv_dir):
        print(f"Directory '{csv_dir}' not found. Creating it and generating file lists.")
        os.makedirs(csv_dir)
        make_file_lists.process_files()
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in '{csv_dir}'. Generating file lists.")
        make_file_lists.process_files()
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    selected_files = {'main': None, 'secondary': None}
    # Auto-select if a file starts with 'main_folder'
    main_folder_file = next((f for f in csv_files if f.startswith('main_folder')), None)
    if main_folder_file:
        selected_files['main'] = os.path.join(csv_dir, main_folder_file)
        csv_files.remove(main_folder_file)
        selected_files['secondary'] = os.path.join(csv_dir, csv_files[0]) if csv_files else selected_files['main']
        print("Auto-selected main and secondary files:", selected_files)
        return selected_files['main'], selected_files['secondary']
    if len(csv_files) == 1:
        selected_files['main'] = selected_files['secondary'] = os.path.join(csv_dir, csv_files[0])
        print("Only one CSV file found, defaulting both to:", selected_files['main'])
        return selected_files['main'], selected_files['secondary']
    # Default to the first two if multiple exist.
    selected_files['main'] = os.path.join(csv_dir, csv_files[0])
    selected_files['secondary'] = os.path.join(csv_dir, csv_files[1])
    print("Multiple CSV files found. Defaulting main to", selected_files['main'], "and secondary to",
          selected_files['secondary'])
    return selected_files['main'], selected_files['secondary']


def get_image_names_from_csv(file_path):
    global png_paths_len
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row[1:] for row in csv_reader]
        png_paths_len = len(png_paths)
    return png_paths


def calculate_index(frame_counter):
    scale_ref = 4.0
    frame_scale = scale_ref / frame_duration
    progress = (decimal.Decimal(frame_counter * frame_scale)) % (png_paths_len * 2)
    if progress < png_paths_len:
        index = int(progress.quantize(decimal.Decimal('1.000'), rounding=decimal.ROUND_HALF_UP))
        direction = 1
    else:
        index = int((decimal.Decimal(png_paths_len * 2) - progress).quantize(decimal.Decimal('1.000'),
                                                                             rounding=decimal.ROUND_HALF_UP))
        direction = -1
    # Clamp the index to valid bounds.
    index = max(0, min(index, png_paths_len))
    return index, direction


def init_all(clock_mode):
    """
    Initializes the preset data, image lists, and frame duration.

    If clock_mode is FREE_CLOCK (255), then the video_length is set to the length
    of the main folder (obtained from the CSV) and frame_duration is computed accordingly.
    Otherwise, the interactive preset setup is used.
    """
    csv_main, csv_float = select_img_list_files()
    main_image_paths = get_image_names_from_csv(csv_main)
    float_image_paths = get_image_names_from_csv(csv_float)

    global video_length, frame_duration
    if clock_mode == FREE_CLOCK or clock_mode == 255:
        video_length = len(main_image_paths)
        frame_duration = video_length / png_paths_len
        print("FREE_CLOCK mode: video_length preset set from main folder length =", video_length)
        print("Computed frame_duration =", frame_duration)
    else:
        calculate_frame_duration()

    return csv_main, main_image_paths, float_image_paths


def main():
    # Example: pass the clock_mode to init_all.
    # For Free Clock mode, use FREE_CLOCK (or 255); for other modes, use a different value.
    clock_mode = FREE_CLOCK  # or any other mode as needed
    init_all(clock_mode)
    # Additional processing can follow...


if __name__ == "__main__":
    main()
