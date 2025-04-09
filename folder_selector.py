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
        folder_dictionary['rand_mult'] = random.randint(1, 9)

    if 'rand_start' not in folder_dictionary:
        rm = folder_dictionary['rand_mult']
        folder_dictionary['rand_start'] = 4 * (FPS - (rm * rm // 2)) + 30

    rand_mult = folder_dictionary['rand_mult']
    rand_start = folder_dictionary['rand_start']

    if CLOCK_MODE == 255:  # Free clock
        # "Rest zones" where we return to default folders
        if index <= rand_start:
            float_folder = 0
            main_folder = 0
        else:
            # Background layer (float folder)
            if index % (1+ FPS * rand_mult) == 0:
                float_folder = random.randint(1, float_folder_count - 1)
                #print(float_folder_count,float_folder)
                folder_dictionary['rand_mult'] = random.randint(1, 12)  # update mult

            # Foreground layer (main folder)
            if index % (2 * FPS * rand_mult + 1) == 0:
                main_folder = random.randint(1, main_folder_count - 1)
                folder_dictionary['rand_mult'] = random.randint(1, 9)  # update mult again

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
