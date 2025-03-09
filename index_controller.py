# index_controller.py
import time
import datetime
import random

# Mode Constants (mirroring the display script)
FULLSCREEN_MODE = True
MTC_CLOCK = 0
MIDI_CLOCK = 1
MIXED_CLOCK = 2
CLIENT_MODE = 3
FREE_CLOCK = 255

# Global state variables for the controller
clock_mode = FREE_CLOCK
midi_mode = False
IPS = 30  # images per second (should match display script)

# Global dictionaries for control data and folder selection.
control_data_dictionary = {
    'Note_On': (0, 127, 0),
    'Note_Off': (None, None, None),
    'Modulation': (0, 0),
    'Index_and_Direction': (0, 1),
    'BPM': 120,
}

folder_dictionary = {
    'Main_and_Float_Folders': (0, 8),
}

valid_modes = {
    "MTC_CLOCK": MTC_CLOCK,
    "MIDI_CLOCK": MIDI_CLOCK,
    "MIXED_CLOCK": MIXED_CLOCK,
    "CLIENT_MODE": CLIENT_MODE,
    "FREE_CLOCK": FREE_CLOCK,
}

# Global launch_time for index calculation.
launch_time = None

def set_launch_time(from_birth=False):
    """Initialize the global launch_time variable."""
    global launch_time
    if from_birth:
        fixed_datetime = datetime.datetime(
            1978, 11, 17, 7, 11,
            tzinfo=datetime.timezone(datetime.timedelta(hours=-5))
        )
        launch_time = fixed_datetime.timestamp()
    else:
        launch_time = time.time()

# Initialize launch_time once.
set_launch_time(from_birth=True)

def set_clock_mode(mode=None):
    """
    Set the clock mode from the provided mode or via user prompt.
    This also sets the global midi_mode.
    """
    global clock_mode, midi_mode
    if mode and mode in valid_modes.values():
        clock_mode = mode
    else:
        while True:
            print("Please choose a clock mode from the following options:")
            for i, (mode_name, mode_value) in enumerate(valid_modes.items(), 1):
                print(f"{i}. {mode_name}")
            user_choice = input("Enter the number corresponding to your choice: ")
            if user_choice.isdigit() and 1 <= int(user_choice) <= len(valid_modes):
                clock_mode = list(valid_modes.values())[int(user_choice) - 1]
                print(f"Clock mode has been set to {list(valid_modes.keys())[int(user_choice) - 1]}")
                break
            else:
                print(f"Invalid input: '{user_choice}'. Please try again.")
    midi_mode = True if (clock_mode < CLIENT_MODE) else False
    print("YER", list(valid_modes.keys())[list(valid_modes.values()).index(clock_mode)])

def calculate_free_clock_index(total_images, pingpong=True):
    """
    Calculate the current frame index and direction based on elapsed time.
    Returns a tuple (index, direction).
    """
    elapsed = time.time() - launch_time
    if pingpong:
        period = 2 * (total_images - 1)
        raw_index = int(elapsed * IPS) % period
        if raw_index >= total_images:
            index = period - raw_index
            direction = -1
        else:
            index = raw_index
            direction = 1
    else:
        index = int(elapsed * IPS) % total_images
        direction = 1
    return index, direction

def update_control_data(index, direction, float_folder_count, main_folder_count):
    """
    Update the folder_dictionary based on the current index and direction.
    Returns the updated (main_folder, float_folder).
    """
    rand_mult = random.randint(1, 9)
    rand_start = 8 * (IPS - (rand_mult * rand_mult // 2))
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    if clock_mode == FREE_CLOCK:
        if index <= rand_start * direction or (100 * rand_start < index < 140 * rand_start):
            float_folder = 0
            main_folder = 0
        elif index % (IPS * rand_mult) == 0:
            float_folder = random.randint(0, float_folder_count - 1)
            rand_mult = random.randint(1, 12)
        elif index % (2 * IPS * rand_mult - 1) == 0:
            main_folder = random.randint(0, main_folder_count - 1)
    else:
        note, channel, _ = control_data_dictionary['Note_On']
        modulation, channel = control_data_dictionary['Modulation']
        mod_value = int(modulation / 127 * float_folder_count)
        float_folder = mod_value % float_folder_count
        main_folder = (note % 12) % main_folder_count
    folder_dictionary['Main_and_Float_Folders'] = (main_folder, float_folder)
    return main_folder, float_folder

def update_index_and_folders(float_folder_count, main_folder_count):
    """
    Retrieve the current index and direction based on the current mode.
    For MIDI or CLIENT modes, gets values from their sources.
    In FREE_CLOCK mode, returns the value from control_data_dictionary.
    Then, updates folder_dictionary via update_control_data.
    Returns a tuple (index, direction).
    """
    global control_data_dictionary
    if midi_mode:
        import midi_control  # if needed
        midi_control.process_midi(clock_mode)
        control_data_dictionary = midi_control.midi_data_dictionary
        index, direction = control_data_dictionary['Index_and_Direction']
    elif clock_mode == CLIENT_MODE:
        import index_client  # if needed
        control_data_dictionary = index_client.midi_data_dictionary
        index, direction = control_data_dictionary['Index_and_Direction']
    elif clock_mode == FREE_CLOCK:
        index, direction = control_data_dictionary['Index_and_Direction']
    update_control_data(index, direction, float_folder_count, main_folder_count)
    return index, direction

def print_index_info(total_images, float_folder_count, main_folder_count):
    """
    Print useful debug information about the current index, direction,
    folder selections, and elapsed time.
    """
    index, direction = control_data_dictionary['Index_and_Direction']
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']
    elapsed = time.time() - launch_time
    print(f"Current index: {index} (direction: {direction})")
    print(f"Folder selection - Main: {main_folder}, Float: {float_folder}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
