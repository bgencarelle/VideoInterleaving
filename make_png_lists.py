import random
import os
import re
from itertools import zip_longest
import csv
import inspect
import sys


def parse_folder_locations():
    folder_dict = {}
    csv_path = choose_file()
    check_unequal_png_counts(csv_path)

    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    break

                number, folder = int(row[0]), row[1]
                if folder:
                    folder_dict[number] = folder
    return folder_dict


def choose_file():
    processed_dir = 'foldersProcessed'
    available_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    print("Available CSV files:")
    for i, file in enumerate(available_files):
        print(f"{i + 1}: {file}")

    while True:
        try:
            choice = int(input("Enter the number corresponding to the desired file: ")) - 1
            if 0 <= choice < len(available_files):
                return os.path.join(processed_dir, available_files[choice])
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def check_unequal_png_counts(csv_path):
    if os.path.exists(csv_path):
        png_counts = set()
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    break
                png_counts.add(int(row[3]))  # Column 4 has index 3
                if len(png_counts) > 1:
                    print("error: unequal png count found, please fix before proceeding")
                    sys.exit(0)


def parse_line(line):
    match = re.match(r'(\d+)\. (.+)', line)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def interleave_lists(lists):
    result = []
    for items in zip_longest(*lists, fillvalue=None):
        group = []
        for item in items:
            if item is not None:
                group.append(item)
        result.append(group)
    return result


def get_folder_weights(folder_count, folder_dict):
    while True:
        print(f"\nThere are {folder_count} folders in total")
        folder_weights = []
        for i in range(folder_count):
            folder_name = os.path.basename(folder_dict[sorted(folder_dict.keys())[i]])
            while True:
                prompt = f"Enter the relative weight for files in \"{folder_name}\" " \
                         f"or hit enter for a default of 1.0" \
                         f" (Remaining folders: {folder_count - i - 1}): "
                weight = input(prompt)
                if weight.strip() == "":
                    folder_weights.append(1)
                    break
                else:
                    try:
                        weight = float(weight) / 100
                        if weight < 0:
                            raise ValueError
                        folder_weights.append(weight)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a valid weight.")

            running_sum = sum(folder_weights)
            print(f"Running sum of weights: {running_sum}")

        # Check if the total weight is greater than zero
        total_weight = sum(folder_weights)
        if total_weight > 0:
            break
        else:
            print("Error: The sum of weights must be greater than zero. Please enter the weights again.")

    # Normalize weights to sum to 1
    folder_weights = [weight / total_weight for weight in folder_weights]

    return folder_weights


def verify_weights(sampled_png_files, folder_weights, folder_dict):
    iteration_count = len(sampled_png_files)
    item_count = sum([len(group) for group in sampled_png_files])

    folder_occurrences = {key: 0 for key in folder_dict.keys()}

    for group in sampled_png_files:
        for filepath in group:
            for i, folder_path in folder_dict.items():
                if folder_path in filepath:
                    folder_occurrences[i] += 1
                    break

    folder_percentages = {key: occurrences / item_count * 100 for key, occurrences in folder_occurrences.items()}

    calling_function = inspect.stack()[1].function  # Get the name of the calling function
    output_filename = f"weights_{calling_function}.txt"

    with open(output_filename, "w") as f:
        f.write("\nVerification of weights:\n")
        f.write(f"Total Iterations: {iteration_count}\n")
        f.write(f"Total Items: {item_count}\n")

        for i in folder_dict.keys():
            folder_name = os.path.basename(folder_dict[sorted(folder_dict.keys())[i - 1]])
            f.write(f"Folder {i - 1} ({folder_name}) representation: "
                    f"{folder_percentages[i]:.2f}% "
                    f"(Expected: {folder_weights[sorted(folder_dict.keys()).index(i)] * 100:.2f}%)\n")

    print(f"Verification of weights saved to {output_filename}")


def write_sorted_random(sampled_png_files, output_folder):
    with open(os.path.join(output_folder, 'sorted_random_list.txt'), 'w') as f:
        for group in sampled_png_files:
            for filepath in group:
                f.write(filepath + '\n')
    print("Sorted randomized list written to sorted_random_list.txt")


def write_sorted_multiplier_random(expanded_sampled_png_files, output_folder):
    with open(os.path.join(output_folder, 'sorted_multiplier_random_list.txt'), 'w') as f:
        for group in expanded_sampled_png_files:
            for filepath in group:
                f.write(filepath + '\n')
    print("Sorted multiplier randomized list written to sorted_multiplier_random.txt")


def write_sorted_png_stream(grouped_png_files, output_folder):
    with open(os.path.join(output_folder, 'sorted_png_stream.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        sorted_grouped_png_files = sorted(enumerate(grouped_png_files), key=lambda x: x[1][0])
        # Sort by the number in C1
        for index, group in sorted_grouped_png_files:
            csv_writer.writerow([index] + group)
    print("Non-Weighted Grouped list written to sorted_png_stream.csv")


def weighted_sampling(grouped_png_files, folder_weights, output_folder):
    sampled_png_files = []

    for group in grouped_png_files:
        sampled_group = []
        for i in range(len(group)):
            chosen_index = random.choices(range(len(group)), weights=folder_weights, k=1)[0]
            sampled_group.append(group[chosen_index])
        sampled_png_files.append(sampled_group)

    with open(os.path.join(output_folder, 'weighted_sampling.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        for index, group in enumerate(sampled_png_files):
            csv_writer.writerow([index] + group)

    print("Weighted sampled list written to weighted_sampling.csv")
    return sampled_png_files


def expanded_weighted_sampling(grouped_png_files, folder_weights, total_items, output_folder):
    expanded_sampled_png_files = []

    for group in grouped_png_files:
        expanded_sampled_group = []
        for _ in range(total_items):
            chosen_index = random.choices(range(len(group)), weights=folder_weights, k=1)[0]
            expanded_sampled_group.append(group[chosen_index])
        expanded_sampled_png_files.append(expanded_sampled_group)

    with open(os.path.join(output_folder, 'expanded_weighted_sampling.csv'), 'w', newline='') as f:
        csv_writer = csv.writer(f)
        for index, group in enumerate(expanded_sampled_png_files):
            csv_writer.writerow([index] + group)

    print("Expanded weighted sampled list written to expanded_weighted_sampling.csv")
    return expanded_sampled_png_files


def process_files():
    folder_dict = parse_folder_locations()
    if not folder_dict:
        print("No accessible files in the foldersProcessed directory.")
        sys.exit(1)  # Terminate the script with a non-zero exit code

    sorted_png_files = sort_png_files(folder_dict)
    grouped_png_files = interleave_lists(sorted_png_files)

    output_folder = "generatedPngLists"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ask_generate_csv(grouped_png_files, output_folder)
    ask_create_weighted_file_lists(sorted_png_files, folder_dict, grouped_png_files, output_folder)


def sort_png_files(folder_dict):
    sorted_png_files = []
    for number in sorted(folder_dict.keys()):
        folder = folder_dict[number]
        png_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
        png_files.sort(key=natural_sort_key)
        sorted_png_files.append(png_files)
    return sorted_png_files


def ask_generate_csv(grouped_png_files, output_folder):
    if input("Do you want to generate a CSV with all of the files? (yes/no, default yes): ").lower().strip() not in ["no", "n"]:
        write_sorted_png_stream(grouped_png_files, output_folder)



def ask_create_weighted_file_lists(sorted_png_files, folder_dict, grouped_png_files, output_folder):
    if input("Do you want to create weighted file lists? (yes/no, default yes): ").lower().strip() not in ["n", "no"]:
        folder_weights = get_folder_weights(len(sorted_png_files), folder_dict)
        ask_generate_weighted_csv(grouped_png_files, folder_weights, output_folder, folder_dict)


def ask_generate_weighted_csv(grouped_png_files, folder_weights, output_folder, folder_dict):
    if input("Do you want to generate a weighted csv? (yes/no, default no): ").lower().strip() in ["yes", "y"]:
        sampled_png_files = weighted_sampling(grouped_png_files, folder_weights, output_folder)
        verify_weights(sampled_png_files, folder_weights, folder_dict)
        ask_generate_list(sampled_png_files, output_folder)
        ask_reapply_weighting(grouped_png_files, folder_weights, output_folder, folder_dict)


def ask_generate_list(sampled_png_files, output_folder):
    if input("Do you want to generate this as a list? (yes/no, default default no): ").lower().strip() in ["yes", "y"]:
        write_sorted_random(sampled_png_files, output_folder)


def ask_reapply_weighting(grouped_png_files, folder_weights, output_folder, folder_dict):
    if input("Now we can reapply this weighting and make a new csv, ok? (yes/no, default no): ").lower().strip() in [
            "yes", "y"]:
        counts = get_counts_input()
        expanded_sampled_png_files = expanded_weighted_sampling(grouped_png_files, folder_weights, counts,
                                                                output_folder)
        verify_weights(expanded_sampled_png_files, folder_weights, folder_dict)
        ask_make_list(expanded_sampled_png_files, output_folder)


def get_counts_input():
    while True:
        try:
            counts = int(input("Enter the number of items you want for each step(must be greater than 0): "))
            if counts > 0:
                break
            else:
                print("Count must be greater than 0.")
        except ValueError:
            print("Invalid input. Please enter a valid integer greater than 0.")
    return counts


def ask_make_list(expanded_sampled_png_files, output_folder):
    if input("Do you want to make this list? (yes/no, default no): ").lower().strip() in ["yes", "y"]:
        write_sorted_multiplier_random(expanded_sampled_png_files, output_folder)


if __name__ == "__main__":
    process_files()
