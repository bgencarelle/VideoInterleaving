# this is the main stuff for all the index calculations and csv file generation
import csv
import os
import mido
import decimal
import make_file_lists

png_paths_len = 2221
frame_duration = 4.0
video_length = 9173
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
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow([video_name, video_name_length])


def get_video_length(video_number=0):
    global frame_duration
    presets_folder = "presets"
    csv_file_path = os.path.join(presets_folder, "set_video_length.csv")

    if not os.path.exists(csv_file_path):
        print("Video length file not found, let's add that now")
        calculate_frame_duration(True)
    with open(csv_file_path, mode="r", newline='') as file:
        reader = csv.reader(file)
        video_lengths = list(reader)

        if len(video_lengths) == 1:
            return int(video_lengths[0][1])

        if 0 < video_number <= len(video_lengths):
            return int(video_lengths[video_number - 1][1])

        print("Please choose a video by name from the list:")
        for i, video in enumerate(video_lengths):
            print(f"{i + 1}. {video[0]}")

        while True:
            selected_video = input("Enter the name of the video: ")
            for video in video_lengths:
                if video[0] == selected_video:
                    return int(video[1])
            print("Invalid video name. Please try again.")


def calculate_frame_duration(setup_mode=False):
    global video_length, frame_duration
    if setup_mode:
        while True:
            user_input = input("Would you like to enter the frame count manually? (y/n): ")
            if user_input.lower() != 'n':
                text_mode = True
            else:
                text_mode = False

            if text_mode:
                while True:
                    video_length_input = input("Please specify the length of the video in frames or type 'midi' to switch mode: ")
                    if video_length_input.lower() == "midi":
                        text_mode = False
                        break
                    try:
                        video_length = int(video_length_input)
                        video_length = abs(video_length)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a positive integer.")
                if text_mode:
                    video_name = input("Please enter the name of the video: ")
                    set_video_length(video_name, video_length)
                    break
            else:
                while True:
                    midi_file_path = input("Please enter the MIDI file path so we can calculate the frame count or type 'text' to switch mode: ")
                    if midi_file_path.lower() == "text":
                        text_mode = True
                        break
                    if not os.path.exists(midi_file_path):
                        print("MIDI file not found. Please try again.")
                    elif not midi_file_path.lower().endswith('.mid'):
                        print("Invalid file type. Please provide a valid MIDI file.")
                    else:
                        try:
                            video_length = get_midi_length(midi_file_path)
                            break
                        except Exception as e:
                            print(f"An error occurred: {e}")
                if not text_mode:
                    video_name = input("Please enter the name of the video: ")
                    set_video_length(video_name, video_length)
                    break

    else:
        video_length = get_video_length()
    frame_duration = video_length / png_paths_len
    print("Frame scaling factor for this video: ", frame_duration)
    return frame_duration


def select_img_list_files():
    csv_dir = 'generated_img_lists'

    while True:
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

        if not csv_files:
            print(f"No .csv files found in directory {csv_dir}")
            make_file_lists.process_files()
            continue

        selected_files = {'main': '', 'secondary': ''}

        # If there is a file that starts with 'main_folder', select it as the main file automatically
        main_folder_file = next((f for f in csv_files if f.startswith('main_folder')), None)
        if main_folder_file:
            selected_files['main'] = os.path.join(csv_dir, main_folder_file)
            csv_files.remove(main_folder_file)
            
            if csv_files:
                # If there's another file, assign it as secondary
                selected_files['secondary'] = os.path.join(csv_dir, csv_files[0])
            else:
                # If there's only one file, use it as both main and secondary
                selected_files['secondary'] = selected_files['main']

            print("Auto-selected main and secondary files:")
            print(selected_files)
            return selected_files['main'], selected_files['secondary']

        if len(csv_files) == 1:
            print("Only one .csv file found, defaulting to main and secondary:")
            selected_files['main'] = selected_files['secondary'] = os.path.join(csv_dir, csv_files[0])
            print(selected_files)
            return selected_files['main'], selected_files['secondary']

        for file_type in ['main', 'secondary']:
            if len(csv_files) == 2 and file_type == 'secondary':
                remaining_file = list(set(csv_files) - set(os.path.basename(fp) for fp in selected_files.values()))[0]
                selected_files[file_type] = os.path.join(csv_dir, remaining_file)
                print(f"Secondary file defaulted to: {selected_files[file_type]}")
                return selected_files['main'], selected_files['secondary']

            print(f"Please select a .csv file to use as {file_type}")
            for i, f in enumerate(csv_files):
                print(f"{i + 1}: {f}")

            while True:
                try:
                    selection = int(input("> "))
                    if selection in range(1, len(csv_files) + 1):
                        selected_files[file_type] = os.path.join(csv_dir, csv_files[selection - 1])
                        csv_files.pop(selection - 1)
                        print(f"Selected {file_type} file: {selected_files[file_type]}")
                        break
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid selection. Please enter a number corresponding to a file.")



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
        current_index, direction = calculate_index(i)

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
        index = int(progress.quantize(decimal.Decimal('1.000'), rounding=decimal.ROUND_HALF_UP))
        direction = 1
    else:
        index = int((decimal.Decimal(png_paths_len * 2) - progress).quantize(decimal.Decimal('1.000'),
                                                                             rounding=decimal.ROUND_HALF_UP))
        direction = -1

    # Ensure the index is within the valid range
    index = max(0, min(index, png_paths_len))

    #print(index * direction)
    return index, direction


def init_all(setup=False):
    global frame_duration
    csv_main, csv_float = select_img_list_files()
    main_image_paths = get_image_names_from_csv(csv_main)
    float_image_paths = get_image_names_from_csv(csv_float)
    # print(png_paths_len == len(main_folder_path), png_paths_len)
    frame_duration = calculate_frame_duration(setup)

    return csv_main, main_image_paths, float_image_paths


def main():
    init_all(False)
    # print(frame_dur)
    # check_index_differences()


if __name__ == "__main__":
    main()
