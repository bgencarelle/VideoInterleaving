import csv
import calculators
import random
import time
import argparse
from datetime import datetime


def get_image_names_from_csv(file_path):
    global png_paths_len
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row for row in csv_reader]  # Include all rows for consistent indexing
        png_paths_len = len(png_paths)
    return png_paths


def load_images(index, main_folder, float_folder):
    main_image = main_folder_path[index][main_folder]
    float_image = float_folder_path[index][float_folder]
    return main_image, float_image


FPS = 30
PINGPONG = True
run_mode = True
png_paths_len = 0
main_folder_path = []
float_folder_path = []
float_folder_count = 0
main_folder_count = 0

control_data_dictionary = {
    'Index_and_Direction': (0, 1),
}

folder_dictionary = {
    'Main_and_Float_Folders': (0, 0),
}


def update_index_and_folders(index, direction):
    global control_data_dictionary

    if PINGPONG:
        # Reverse direction at boundaries without changing the current index
        if (index + direction) < 0 or (index + direction) >= png_paths_len:
            direction *= -1  # Reverse direction
        else:
            index += direction
    else:
        index = (index + 1) % png_paths_len

    control_data_dictionary['Index_and_Direction'] = (index, direction)
    update_control_data(index, direction)
    return index, direction


def update_control_data(index, direction):
    rand_mult = random.randint(1, 9)
    rand_start = 8 * (FPS - (rand_mult * rand_mult // 2))

    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']

    if index <= rand_start * direction or (index > 100 * rand_start and index < 140 * rand_start):
        float_folder = 0
        main_folder = 0
    elif index % (FPS * rand_mult) == 0:
        float_folder = random.randint(0, float_folder_count - 1)
        rand_mult = random.randint(1, 12)
    elif index % (2 * FPS * rand_mult - 1) == 0:
        main_folder = random.randint(0, main_folder_count - 1)

    folder_dictionary['Main_and_Float_Folders'] = (main_folder, float_folder)


def main_csv_test(cycles):
    output_rows = []  # Initialize the list to hold CSV rows
    global png_paths_len, main_folder_path, main_folder_count, float_folder_path, float_folder_count

    random.seed()
    csv_source, main_folder_path, float_folder_path = calculators.init_all()
    if not (main_folder_path and float_folder_path):
        raise ValueError("Failed to load paths from calculators.init_all(). Ensure the function returns valid paths.")

    png_paths_len = len(main_folder_path)
    main_folder_count = len(main_folder_path[0]) if png_paths_len > 0 else 0
    float_folder_count = len(float_folder_path[0]) if png_paths_len > 0 else 0

    if png_paths_len == 0:
        raise ValueError("No PNG paths found in the main folder.")

    index, direction = 0, 1
    absolute_index = 0

    # Calculate the correct total iterations based on ping-pong logic with duplication at boundaries
    total_iterations = cycles * (2 * png_paths_len) if png_paths_len > 1 else cycles

    for iteration in range(total_iterations):
        # Load current images
        main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
        main_image, float_image = load_images(index, main_folder, float_folder)
        output_rows.append([absolute_index, main_image, float_image])

        # Update index and direction
        prev_index, prev_direction = index, direction
        index, direction = update_index_and_folders(index, direction)

        # Increment absolute_index after appending
        absolute_index += 1

        # Debugging: Uncomment the following line to trace the execution
        # print(f"Iteration: {iteration}, Index: {index}, Direction: {direction}, Absolute Index: {absolute_index}")

    return output_rows


def write_to_csv(output_rows):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"main_float_{png_paths_len}_{timestamp}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Absolute Index', 'Main Image', 'Float Image'])
        writer.writerows(output_rows)


def run(cycles):
    output_rows = main_csv_test(cycles)
    write_to_csv(output_rows)
    print(f"CSV file created with {len(output_rows)} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image paths and generate a CSV.")
    parser.add_argument(
        "--cycles", type=int, default=None, help="Number of full forward and reverse cycles"
    )
    args = parser.parse_args()

    if args.cycles is None:
        while True:
            try:
                cycles = int(input("Enter the number of full forward and reverse cycles: "))
                if cycles <= 0:
                    print("Please enter a positive integer.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a positive integer.")
    else:
        cycles = args.cycles

    run(cycles)
