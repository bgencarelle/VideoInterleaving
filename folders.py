import os

def get_folder_paths():
    # Create folder_locations.txt if it doesn't exist
    if not os.path.exists('folder_locations.txt'):
        with open('folder_locations.txt', 'w'):
            pass

    # Read existing folder locations from folder_locations.txt
    with open('folder_locations.txt', 'r') as f:
        folders = f.read().splitlines()

    # Prompt the user to add more folders
    new_folders = []
    while True:
        user_input = input("Enter a folder path to add (or leave blank to finish): ")
        if user_input == "":
            break
        elif os.path.isdir(user_input):
            folders.append(user_input)
            new_folders.append(user_input)
        else:
            print("Invalid folder path.")

    # Save the new folder locations to folder_locations.txt
    with open('folder_locations.txt', 'a') as f:
        for folder in new_folders:
            f.write(f"{folder}\n")

    # Get a list of all the PNG files in each folder and its immediate subdirectories, sorted alphabetically
    folder_files = []
    for folder in folders:
        files = []
        for dirpath, dirnames, filenames in os.walk(folder):
            for file in filenames:
                if file.lower().endswith('.png'):
                    files.append(os.path.join(dirpath, file))
            # Add the subdirectories to the list and sort
            for subdir in dirnames:
                folders.append(os.path.join(dirpath, subdir))
            # Break the loop to avoid going deeper than one level
            break
        files.sort()
        folder_files.append(files)

    # Merge the files interleaved alphabetically
    merged_files = []
    while any(folder_files):
        for i in range(len(folder_files)):
            if folder_files[i]:
                merged_files.append(folder_files[i].pop(0))

    # Write the merged file list to sorted.txt
    with open('sorted.txt', 'w') as f:
        for file_path in merged_files:
            f.write(f"{file_path}\n")

    print("PNG file list written to sorted.txt")

    # Call the function to get folder paths and write the PNG file list to sorted.txt
    get_folder_paths()
