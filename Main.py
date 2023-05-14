import os
from get_folders_list import write_folder_list
from make_file_lists import process_files
from image_display import display_and_run

def main():

    if input("Do you want to run_mode write_folder_list()? (y/n) ").lower() == 'y':
        try:
            write_folder_list()
        except Exception as e:
            print(f"Error: {e}")
            main()
            return  # Exit the script if an exception occurs in write_folder_list

    if input("Do you want to run_mode process_files()? (y/n) ").lower() == 'y':
        print("there are a few things we can do here. Let's go through the options")
        try:
            process_files()
        except Exception as e:
            print(f"Error: {e}: \n Restarting from the beginning because you like to waste time on nonsense")
            main()
            return  # Exit the script if an exception occurs in process_files

    if not input("Do you want to run_mode image_display()? (y/n) ").lower() == 'n':
        csv_files = [f for f in os.listdir('generatedPngLists') if f.endswith('.csv')]
        if csv_files:
            display_and_run()
        else:
            print("Cannot run_mode mtc_png_realtime_midi(). No .csv files found in directory generatedPngLists.")
            print("well that was a waste of time")
            main()
    else:
        return


if __name__ == "__main__":
    main()
