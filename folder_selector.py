import math
import random
from globals import control_data_dictionary, folder_dictionary
from settings import (FPS, CLOCK_MODE)
from utilities.csv_list_maker import main_folder_path


def update_folder_selection(index, float_folder_count, main_folder_count):
    """
    Maintains persistent random timing for folder switching.
    Folders reset to 0 in rest zones and switch based on randomized modulus logic.
    """
    # Unpack last used folders
    main_folder, float_folder = folder_dictionary.get('Main_and_Float_Folders', (0, 0))

    # Initialize persistent random state
    if 'rand_mult' not in folder_dictionary:
        random.seed()
        folder_dictionary['rand_mult'] = random.randint(1, 9)

    if 'rand_start' not in folder_dictionary:
        rm = folder_dictionary['rand_mult']
        folder_dictionary['rand_start'] = random.randint(FPS, int(3.5 * FPS))

    rand_mult = folder_dictionary['rand_mult']
    rand_start = folder_dictionary['rand_start']

    if CLOCK_MODE == 255:  # Free clock
        # Ensure the active cycle flag exists; default it to False.
        if 'active_cycle' not in folder_dictionary:
            folder_dictionary['active_cycle'] = False

        if index <= rand_start:
            # "Rest zone": keep both folders at 0 and reset the active cycle flag.
            float_folder = 0
            main_folder = 0
            folder_dictionary['active_cycle'] = False
        elif not folder_dictionary['active_cycle']:
            # Just after the rest zone (first active frame): update both folders once.
            float_folder = random.randint(1, float_folder_count - 1)
            main_folder = random.randint(1, main_folder_count - 1)
            # Update the random parameter for the next cycle.
            folder_dictionary['rand_mult'] = random.randint(4, 12)
            folder_dictionary['active_cycle'] = True
        else:
            # Active phase: perform periodic updates.
            if index % (((rand_mult // 3) + FPS) * rand_mult) == 7:
                float_folder = random.randint(0, float_folder_count - 1)
                folder_dictionary['rand_mult'] = random.randint(1, 12)  # update mult for float_folder
            if index % (FPS * (rand_mult + rand_mult // 2)) == 3:
                main_folder = random.randint(1, main_folder_count - 1)
                folder_dictionary['rand_mult'] = random.randint(1, 9)  # update mult for main_folder
                folder_dictionary['rand_start'] = random.randint(FPS, int(3.5 * FPS))



    else:
        # MIDI-driven case (unchanged logic)
        note, channel, _ = control_data_dictionary['Note_On']
        modulation, channel = control_data_dictionary['Modulation']
        mod_value = int(modulation / 127 * float_folder_count)
        float_folder = mod_value % float_folder_count
        main_folder = (note % 12) % main_folder_count

    # Save updated folder selections
    folder_dictionary['Main_and_Float_Folders'] = (main_folder, float_folder)
    return main_folder, float_folder
