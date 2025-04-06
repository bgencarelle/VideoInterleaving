#folder_selector.py
import random
from globals import control_data_dictionary, folder_dictionary
from settings import (FPS, CLOCK_MODE)

def update_folder_selection(index, direction, float_folder_count, main_folder_count):
    """
    Updated version of update_folder_selection modeled more closely on the JavaScript logic.
    """
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']

    # Initialize random parameters similar to the JS version.
    rand_mult = random.randint(1, 9)

    if CLOCK_MODE == 255:
        if (index < FPS * rand_mult // 3) or (index > 20 * rand_mult and index < 21 * rand_mult):
            float_folder = 0
            main_folder = 0
        else:
            if index % (FPS * rand_mult) == 0:
                float_folder = random.randint(0, float_folder_count - 1)
                #print(float_folder_count)
                # Reassign rand_mult to a new random value (1 to 12) after a float folder update.
                rand_mult = random.randint(1, 12)
            # Random main folder update:
            if index % (2 + FPS * rand_mult + 1) == 0:
                main_folder = random.randint(0, main_folder_count - 1)
                rand_mult = random.randint(1, 9)
    else:
        # Note-based approach (unchanged):
        note, channel, _ = control_data_dictionary['Note_On']
        modulation, channel = control_data_dictionary['Modulation']
        mod_value = int(modulation / 127 * float_folder_count)
        float_folder = mod_value % float_folder_count
        main_folder = (note % 12) % main_folder_count

    folder_dictionary['Main_and_Float_Folders'] = (main_folder, float_folder)
    #print(folder_dictionary)
    return main_folder, float_folder
