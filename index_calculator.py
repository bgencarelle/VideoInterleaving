import time
import datetime
import math
from globals import control_data_dictionary
from settings import IPS, CLIENT_MODE, VALID_MODES, FROM_BIRTH, CLOCK_MODE

clock_mode = CLOCK_MODE
midi_mode = False
launch_time = 0.00000000

# --- Helper for timezone offsets ---
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
    """
    Calculates the current index based on the absolute time elapsed since launch.
    This method produces an index that is fully determined by elapsed time,
    ensuring that different devices started at the same moment will be synchronized.

    For pingpong mode, we set:
      period = 2 * total_images,
      and fold the index as follows:
        If mod_index < total_images:
          index = mod_index
        Else:
          index = (2 * total_images - 1) - mod_index

    This ensures both endpoints (0 and total_images-1) each get repeated once:
      0,1,2,...,N-1, N-1,N-2,...,1,0,0,1,2,...
    """
    elapsed = time.time() - launch_time
    raw_index = math.floor(elapsed * IPS)

    if pingpong:
        # If total_images < 2, there's nothing to ping-pong.
        if total_images < 2:
            index = 0
        else:
            # New period to ensure both endpoints repeat.
            period = 2 * total_images
            mod_index = raw_index % period
            if mod_index < total_images:
                index = mod_index
            else:
                index = (2 * total_images - 1) - mod_index
    else:
        index = raw_index % total_images

    # No need for a 'direction' value; we store None for compatibility.
    control_data_dictionary['Index_and_Direction'] = (index, None)
    return index, None

def update_index(total_images, pingpong=True):
    """
    Update the index using MIDI data if in MIDI mode; otherwise use the free-clock calculation.
    """
    global control_data_dictionary, clock_mode, midi_mode
    if midi_mode:
        import midi_control
        midi_control.process_midi(clock_mode)
        control_data_dictionary.update(midi_control.midi_data_dictionary)
        index, _ = control_data_dictionary['Index_and_Direction']
        return index, None
    elif clock_mode == CLIENT_MODE:
        import index_client
        control_data_dictionary.update(index_client.midi_data_dictionary)
        index, _ = control_data_dictionary['Index_and_Direction']
        return index, None
    else:
        return calculate_free_clock_index(total_images, pingpong)
