import os
from get_folders_list import write_folder_list
from make_png_lists import process_files
from midi_mtc_png_display import mtc_png_realtime


def main():

    if input("Do you want to run write_folder_list()? (y/n) ").lower() == 'y':
        try:
            write_folder_list()
        except Exception as e:
            print(f"Error: {e}")
            main()
            return  # Exit the script if an exception occurs in write_folder_list

    if input("Do you want to run process_files()? (y/n) ").lower() == 'y':
        print("there are a few things we can do here. Let's go through the options")
        try:
            process_files()
        except Exception as e:
            print(f"Error: {e}: \n Restarting from the beginning because you like to waste time on nonsense")
            main()
            return  # Exit the script if an exception occurs in process_files

    if input("Do you want to run midi_mtc_png_display()? (y/n) ").lower() == 'y':
        csv_files = [f for f in os.listdir('generatedPngLists') if f.endswith('.csv')]
        if csv_files:
            mtc_png_realtime()
        else:
            print("Cannot run mtc_png_realtime_midi(). No .csv files found in directory generatedPngLists.")
            print("well that was a waste of time")
            main()
    else:
        return


if __name__ == "__main__":
    main()
