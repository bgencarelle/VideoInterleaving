# index_calculator.py
import time
import datetime
import math
import decimal
from zoneinfo import ZoneInfo

from globals import control_data_dictionary
from settings import IPS, CLIENT_MODE, VALID_MODES, FROM_BIRTH, CLOCK_MODE, BIRTH_TZ, BIRTH_TIME, TIMEZONE_OFFSETS

clock_mode = CLOCK_MODE
midi_mode = False
# midi_control is bound by set_clock_mode(), not imported here.  It pulls in
# `mido` at module level, so importing it here made a MIDI stack a hard
# requirement of every mode -- including a free-running clock that never reads
# a MIDI message, and scope mode on a box with no MIDI at all.  Deferring it
# also breaks the import cycle: midi_control imports this module back, and by
# the time set_clock_mode runs, this one has finished loading.
midi_control = None
#import index_client
launch_time = 0  # Stored as nanoseconds (integer)
# Module-level variables for MIDI clock modes (set by make_file_lists.initialize_image_lists)
png_paths_len = 0
frame_duration = 1.0

def get_timezone(tz_str):
    """Return an IANA timezone using the platform timezone database."""
    return ZoneInfo(tz_str)

def set_launch_time(from_birth=False):
    global launch_time
    birth_year, birth_month, birth_day, birth_hour, birth_minute = map(int, BIRTH_TIME.split(", "))
    if from_birth:
        tz = get_timezone(BIRTH_TZ)
        fixed_datetime = datetime.datetime(birth_year, birth_month, birth_day, birth_hour, birth_minute, tzinfo=tz)
        # Convert timestamp (seconds) to nanoseconds for consistency
        launch_time = int(fixed_datetime.timestamp() * 1_000_000_000)
    else:
        launch_time = time.time_ns()

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
    if midi_mode:
        # Bind it HERE, at startup, not inside update_index. update_index runs
        # once per trace, so an import statement there paid the full mido +
        # rtmidi backend load inside the render loop on the first MIDI frame --
        # a guaranteed dropped trace -- and, with mido missing, raised ~30x a
        # second from the render loop instead of once, clearly, at startup.
        global midi_control
        import midi_control
    print("Clock mode set to", list(VALID_MODES.keys())[list(VALID_MODES.values()).index(clock_mode)])

def calculate_free_clock_index(total_images, pingpong=True, *, time_offset_ns=0, at_time_ns=None, publish=True):
    """
    Fast, mirrored ping‑pong index:
      0,1,2,...,N-1, N-1,N-2,...,1,0, 0,1,2...

    Uses nanosecond precision for tighter synchronization, especially important
    for multi-machine setups with chrony-synchronized clocks.
    """
    # Use nanosecond precision with integer arithmetic to avoid floating point errors
    # Output adapters can select ahead without changing the shared epoch or IPS.
    # Existing callers retain precisely the same clock behavior (offset zero).
    current_time_ns = (time.time_ns() if at_time_ns is None else int(at_time_ns)) + int(time_offset_ns)
    elapsed_ns = current_time_ns - launch_time
    # Calculate index using integer math: (elapsed_ns * IPS) // 1_000_000_000
    raw_index = (elapsed_ns * IPS) // 1_000_000_000
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

    if publish:
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

def update_index(total_images, pingpong=True, *, time_offset_ns=0, at_time_ns=None):
    """
    Update the index using MIDI data if in MIDI mode; otherwise use the free-clock calculation.
    """
    global control_data_dictionary, clock_mode, midi_mode
    if midi_mode and midi_control is None:
        # set_clock_mode() binds it; reaching here without that means a MIDI
        # clock was selected some other way. Say so once and keep drawing on
        # the free clock rather than raising once per trace from the loop.
        print("[CLOCK] MIDI mode requested but midi_control was never loaded; "
              "falling back to the free clock.")
        midi_mode = False
    if midi_mode:
        midi_control.process_midi(clock_mode)
        control_data_dictionary.update(midi_control.midi_data_dictionary)
        index, _ = control_data_dictionary['Index_and_Direction']
        return index, None
    else:
        return calculate_free_clock_index(total_images, pingpong,
                                          time_offset_ns=time_offset_ns, at_time_ns=at_time_ns)
