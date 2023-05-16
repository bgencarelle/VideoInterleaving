import os
from get_folders_list import write_folder_list
from make_file_lists import process_files
from image_display import display_and_run

def main():

    while True:
        user_input = input("Do you want to run_mode write_folder_list()? (y/n) ").lower()
        if user_input == 'y':
            try:
                write_folder_list()
            except Exception as e:
                print(f"Error: {e}")
                main()
                return  # Exit the script if an exception occurs in write_folder_list
            break
        elif user_input == 'n':
            break
        else:
            print("Please enter 'y' or 'n'.")

    while True:
        user_input = input("Do you want to run_mode process_files()? (y/n) ").lower()
        if user_input == 'y':
            print("There are a few things we can do here. Let's go through the options")
            try:
                process_files()
            except Exception as e:
                print(f"Error: {e}: \n Restarting from the beginning because you like to waste time on nonsense")
                main()
                return  # Exit the script if an exception occurs in process_files
            break
        elif user_input == 'n':
            break
        else:
            print("Please enter 'y' or 'n'.")

    while True:
        user_input = input("Do you want to run_mode image_display()? (y/n) ").lower()
        if user_input == 'y':
            csv_files = [f for f in os.listdir('generatedIMGLists') if f.endswith('.csv')]
            if csv_files:
                display_and_run()
            else:
                print("Cannot run_mode mtc_png_realtime_midi(). No .csv files found in directory generatedPngLists.")
                print("Well, that was a waste of time")
                main()
            break
        elif user_input == 'n':
            return
        else:
            print("Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()
