import random
from globals import control_data_dictionary, folder_dictionary
from settings import (IPS, CLOCK_MODE)


def update_folder_selection(index, direction, float_folder_count, main_folder_count):
    rand_mult = random.randint(1, 9)
    rand_start = 8 * (IPS - (rand_mult * rand_mult // 2))
    main_folder, float_folder = folder_dictionary['Main_and_Float_Folders']

    if CLOCK_MODE  == 255:
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
