# index_calculator.py
import time
import datetime
import math
import decimal
from globals import control_data_dictionary
from settings import IPS, CLIENT_MODE, VALID_MODES, FROM_BIRTH, CLOCK_MODE, BIRTH_TZ, BIRTH_TIME, TIMEZONE_OFFSETS

clock_mode = CLOCK_MODE
midi_mode = False
import midi_control
#import index_client
launch_time = 0.00000000

# Module-level variables for MIDI clock modes (set by make_file_lists.initialize_image_lists)
png_paths_len = 0
frame_duration = 1.0


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
    birth_year, birth_month, birth_day, birth_hour, birth_minute = map(int, BIRTH_TIME.split(", "))
    if from_birth:
        tz = get_timezone(BIRTH_TZ)
        fixed_datetime = datetime.datetime(birth_year, birth_month, birth_day, birth_hour, birth_minute, tzinfo=tz)
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
    Fast, mirrored ping‑pong index:
      0,1,2,...,N-1, N-1,N-2,...,1,0, 0,1,2...
    """
    # 1) Replace Decimal + math.floor with a simple int() cast
    raw_index = int((time.time() - launch_time) * IPS)

    if pingpong and total_images > 1:
        period    = 2 * total_images
        mod_index = raw_index % period

        if mod_index < total_images:
            # forward ramp: 0 → N-1
            index = mod_index
        else:
            # mirrored ramp with double‑pivot at both ends
            index = (period - 1) - mod_index
    else:
        index = raw_index % total_images if total_images > 0 else 0

    control_data_dictionary['Index_and_Direction'] = (index, None)
    return index, None

def calculate_midi_clock_index(frame_counter, png_paths_len_param=None, frame_duration_param=None):
    """
    Calculate index from frame counter for MIDI clock modes.
    Uses module-level variables if parameters not provided (for backward compatibility).
    """
    # Use parameters if provided, otherwise fall back to module-level variables
    png_len = png_paths_len_param if png_paths_len_param is not None else png_paths_len
    frame_dur = frame_duration_param if frame_duration_param is not None else frame_duration
    
    scale_ref = 4.0
    frame_scale = scale_ref / frame_dur
    progress = (decimal.Decimal(frame_counter * frame_scale)) % (png_len * 2)
    if progress < png_len:
        index = int(progress.quantize(decimal.Decimal('1.000'), rounding=decimal.ROUND_HALF_UP))
        direction = 1
    else:
        index = int((decimal.Decimal(png_len * 2) - progress).quantize(decimal.Decimal('1.000'),
                                                                         rounding=decimal.ROUND_HALF_UP))
        direction = -1
    index = max(0, min(index, png_len))
    return index, direction


def update_index(total_images, pingpong=True):
    """
    Update the index using MIDI data if in MIDI mode; otherwise use the free-clock calculation.
    """
    global control_data_dictionary, clock_mode, midi_mode
    if midi_mode:
        midi_control.process_midi(clock_mode)
        control_data_dictionary.update(midi_control.midi_data_dictionary)
        index, _ = control_data_dictionary['Index_and_Direction']
        return index, None
    else:
        return calculate_free_clock_index(total_images, pingpong)
