# globals.py

from settings import MAIN_FOLDER_PATH, FLOAT_FOLDER_PATH
control_data_dictionary = {
    'Note_On': (0, 127, 0),
    'Note_Off': (None, None, None),
    'Modulation': (0, 0),
    'Index_and_Direction': (0, 1),
    'BPM': 120,
}

folder_dictionary = {
    'Main_and_Float_Folders': (0, 0),
}

midi_data_dictionary = {
    'Note_On': (0, 127, 0),
    'Note_Off': (None, None, None),
    'Modulation': (0, 0),
    'Index_and_Direction': (0, 1),
    'BPM': (120),
    # 'Stop': False,
    # 'Start': False,
    # 'Pause': False,
    # 'Reset': False
}
