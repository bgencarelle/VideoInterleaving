import time
import datetime
from globals import control_data_dictionary
from settings import IPS, MTC_CLOCK, MIDI_CLOCK, MIXED_CLOCK, CLIENT_MODE, FREE_CLOCK, VALID_MODES, FROM_BIRTH, CLOCK_MODE

clock_mode = CLOCK_MODE
midi_mode = False
launch_time = None

# --- New Helper Function ---
# A simple mapping for common timezone abbreviations.
# Feel free to extend this dictionary as needed.
TIMEZONE_OFFSETS = {
    "EST": -5,
    "EDT": -4,
    "CST": -6,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
}

def get_timezone(tz_str):
    """
    Returns a datetime.timezone object based on the given timezone abbreviation.
    For example, "EST" returns UTC-5.
    """
    if tz_str not in TIMEZONE_OFFSETS:
        raise ValueError(f"Unknown timezone abbreviation: {tz_str}")
    offset_hours = TIMEZONE_OFFSETS[tz_str]
    return datetime.timezone(datetime.timedelta(hours=offset_hours))

def set_launch_time(from_birth=False):
    global launch_time
    if from_birth:
        # Here, you can modify the timezone string ("EST") as needed.
        tz = get_timezone("EST")
        fixed_datetime = datetime.datetime(1978, 11, 17, 7, 11, tzinfo=tz)
        launch_time = fixed_datetime.timestamp()
    else:
        launch_time = time.time()

set_launch_time(from_birth=FROM_BIRTH)

def set_clock_mode(mode=None):
    global clock_mode, midi_mode
    if mode and mode in VALID_MODES.values():
        clock_mode = mode
    else:
        while True:
            print("Please choose a clock mode from the following options:")
            for i, (mode_name, mode_value) in enumerate(VALID_MODES.items(), 1):
                print(f"{i}. {mode_name}")
            user_choice = input("Enter the number corresponding to your choice: ")
            if user_choice.isdigit() and 1 <= int(user_choice) <= len(VALID_MODES):
                clock_mode = list(VALID_MODES.values())[int(user_choice) - 1]
                print(f"Clock mode has been set to {list(VALID_MODES.keys())[int(user_choice) - 1]}")
                break
            else:
                print(f"Invalid input: '{user_choice}'. Please try again.")
    midi_mode = True if (clock_mode < CLIENT_MODE) else False
    print("Clock mode set to", list(VALID_MODES.keys())[list(VALID_MODES.values()).index(clock_mode)])

def calculate_free_clock_index(total_images, pingpong=True):
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
    control_data_dictionary['Index_and_Direction'] = (index, direction)
    return index, direction

def update_index(total_images, pingpong=True):
    """
    Update the index using MIDI data if in MIDI mode, otherwise use the free-clock calculation.
    For MIDI mode, this function calls midi_control.process_midi (or index_client for CLIENT_MODE)
    and updates the shared control_data_dictionary.
    """
    global control_data_dictionary, clock_mode, midi_mode
    if midi_mode:
        import midi_control
        midi_control.process_midi(clock_mode)
        # Assume that midi_control.midi_data_dictionary is updated with the latest MIDI index/direction.
        control_data_dictionary.update(midi_control.midi_data_dictionary)
        index, direction = control_data_dictionary['Index_and_Direction']
        return index, direction
    elif clock_mode == CLIENT_MODE:
        import index_client
        control_data_dictionary.update(index_client.midi_data_dictionary)
        index, direction = control_data_dictionary['Index_and_Direction']
        return index, direction
    else:
        return calculate_free_clock_index(total_images, pingpong)
