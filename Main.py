import os
import get_folders_list
import calculators
import image_display
import make_file_lists


def main(setup_mode=True):

    if setup_mode:
        while True:
            user_input = input("shall we choose new image directories? (y/n) ").lower()
            if user_input != 'n':
                try:
                    get_folders_list.write_folder_list()
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
            user_input = input("Do you want to create our image lists? (y/n) ").lower()
            if user_input != 'n':
                print("There are a few things we can do here. Let's go through the options")
                try:
                    make_file_lists.process_files()
                except Exception as e:
                    print(f"Error: {e}: \n Restarting from the beginning, something went wrong")
                    main()
                    return  # Exit the script if an exception occurs in process_files
                break
            elif user_input == 'n':
                break
            else:
                print("Please enter 'y' or 'n'.")

        while True:
            user_input = input("Do you want to calculate the playback ratio? (y/n) ").lower()
            if user_input != 'n':
                print("ok, you can determine this manually or from a midi file")
                try:
                    calculators.init_all(True)
                except Exception as e:
                    print(f"Error: {e}: \n Restarting from the beginning, something went wrong")
                    main()
                    return  # Exit the script if an exception occurs in process_files
                break
            elif user_input == 'n':
                break
            else:
                print("Please enter 'y' or 'n'.")

        while True:
            user_input = input("Do you want to run our main image_display()? (y/n) ").lower()
            if user_input != 'n':
                csv_files = [f for f in os.listdir('generatedIMGLists') if f.endswith('.csv')]
                if csv_files:
                    image_display.display_and_run()
                else:
                    print("Error: No .csv files found in directory generatedIMGLists.")
                    main()
                break
            elif user_input == 'n':
                return
            else:
                print("Please enter 'y' or 'n'.")
    else:
        image_display.display_and_run()


if __name__ == "__main__":
    main(True)
