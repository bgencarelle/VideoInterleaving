import os

def find_missing_files(folder_path):
    all_files = sorted(os.listdir(folder_path))
    missing_files = []

    for i in range(len(all_files) - 1):
        current_file = all_files[i]
        next_file = all_files[i + 1]

        current_file_name, current_file_ext = os.path.splitext(current_file)
        next_file_name, next_file_ext = os.path.splitext(next_file)

        if current_file_ext != next_file_ext:
            continue

        current_file_number = int(''.join(filter(str.isdigit, current_file_name)))
        next_file_number = int(''.join(filter(str.isdigit, next_file_name)))

        if next_file_number - current_file_number > 1:
            missing_range = list(range(current_file_number + 1, next_file_number))
            for number in missing_range:
                missing_file = current_file_name.rstrip('0123456789') + str(number) + current_file_ext
                missing_files.append(missing_file)

    return missing_files

while True:
    folder_path = input("Enter the folder path: ").strip()

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        missing_files = find_missing_files(folder_path)

        if missing_files:
            print("Missing files:")
            for missing_file in missing_files:
                print(missing_file)
        else:
            print("No files are missing.")
        break
    else:
        print("Invalid directory. Please try again.")
