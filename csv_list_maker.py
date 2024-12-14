import csv
import calculators
import random
import time


def get_image_names_from_csv(file_path):
    global png_paths_len
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        png_paths = [row for row in
                     csv_reader]  # Include all rows for consistent indexing  # Include all columns for consistent indexing
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
        # Reverse direction at boundaries
        if (index + direction) < 0 or index + direction >= png_paths_len:
            direction *= -1
        index += direction
    else:
        index = (index + 1) % png_paths_len

    control_data_dictionary['Index_and_Direction'] = index, direction
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

    folder_dictionary['Main_and_Float_Folders'] = main_folder, float_folder


def main_csv_test(cycles):
    output_rows = []  # Initialize the list to hold CSV rows
    global png_paths_len, main_folder_path, main_folder_count, float_folder_path, float_folder_count

    random.seed()
    csv_source, main_folder_path, float_folder_path = calculators.init_all()
    if not (main_folder_path and float_folder_path):
        raise ValueError("Failed to load paths from calculators.init_all(). Ensure the function returns valid paths.")

    png_paths_len = len(main_folder_path)
    main_folder_count = len(main_folder_path[0])
    float_folder_count = len(float_folder_path[0])

    index, direction = 0, 1
    absolute_index = 0  # Start absolute index at 0

    for _ in range(cycles * 2 * png_paths_len):  # Full forward and reverse cycle
        main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
        main_image, float_image = load_images(index, main_folder, float_folder)
        output_rows.append([absolute_index, main_image, float_image])
        index, direction = update_index_and_folders(index, direction)
        main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
        main_image, float_image = load_images(index, main_folder, float_folder)
        output_rows.append([absolute_index, main_image, float_image])

        absolute_index += 1

    return output_rows


def write_to_csv(output_rows):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"main_float_{png_paths_len}_{timestamp}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Absolute Index', 'Main Image', 'Float Image'])
        writer.writerows(output_rows)


if __name__ == "__main__":
    while True:
        try:
            cycles = int(input("Enter the number of full forward and reverse cycles: "))
            if cycles <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
    output_rows = main_csv_test(cycles)
    write_to_csv(output_rows)
