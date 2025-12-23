import math
import random
import time
from collections import deque

#from rtmidi.midiconstants import LOCAL_CONTROL

from globals import control_data_dictionary, folder_dictionary
from settings import CLOCK_MODE, IPS
LOCAL_CONTROLFPS = 2*IPS #local fps


def update_folder_selection(index, float_folder_count, main_folder_count, folder_dict=None):
    """
    Maintains persistent random timing for folder switching.
    Folders reset to 0 in rest zones and switch based on randomized modulus logic.
    Uses a single RNG and pre-rolled sequences for improved entropy,
    while preserving the rest → first → periodic structure.
    Enforces a minimum ~1 s gap, caps at ~20 s, and adds slight random jitter.
    Timing is based on real elapsed time, not ping-pong index.
    """
    # Use provided dictionary or fallback to global
    if folder_dict is None:
        folder_dict = folder_dictionary
    
    # Retrieve last folders
    main_folder, float_folder = folder_dict.get(
        'Main_and_Float_Folders', (0, 0)
    )

    # SINGLE RNG instance (per dictionary)
    if 'rng' not in folder_dict:
        folder_dict['rng'] = random.Random()
        folder_dict['rng'].seed(time.time())
    rng = folder_dict['rng']

    # Initialize last-pick times for spacing enforcement
    now = time.time()
    if 'last_main_time' not in folder_dict:
        folder_dict['last_main_time'] = now
    if 'last_float_time' not in folder_dict:
        folder_dict['last_float_time'] = now

    # PRE-ROLL sequences for main and float (skip folder 0 for "boring" main)
    if 'pre_main' not in folder_dict:
        seq = list(range(1, main_folder_count))
        folder_dict['pre_main'] = deque()
        for _ in range(2):
            rng.shuffle(seq)
            folder_dict['pre_main'].extend(seq)
    if 'pre_float' not in folder_dict:
        seq = list(range(0, float_folder_count))
        folder_dict['pre_float'] = deque()
        for _ in range(2):
            rng.shuffle(seq)
            folder_dict['pre_float'].extend(seq)

    # Timing parameters via single RNG
    if 'rand_mult' not in folder_dict:
        folder_dict['rand_mult'] = rng.randint(1, 9)
    if 'rand_start' not in folder_dict:
        # rest duration in frames
        folder_dict['rand_start'] = rng.randint(LOCAL_CONTROLFPS, int(3.5 * LOCAL_CONTROLFPS))

    rand_mult = folder_dict['rand_mult']
    rand_start = folder_dict['rand_start']

    # Active-cycle flag
    if 'active_cycle' not in folder_dict:
        folder_dict['active_cycle'] = False

    if CLOCK_MODE == 255:
        # Rest zone: both folders = 0
        if index <= rand_start:
            if folder_dict['active_cycle']:
                # On entering rest, reroll next start
                folder_dict['rand_start'] = rng.randint(LOCAL_CONTROLFPS, int(3.5 * LOCAL_CONTROLFPS))
            folder_dict['active_cycle'] = False
            main_folder = 0
            float_folder = 0
        # First active pick
        elif not folder_dict['active_cycle']:
            # Pop initial values
            float_folder = folder_dict['pre_float'].popleft()
            main_folder  = folder_dict['pre_main'].popleft()
            # New tempo for next beats
            folder_dict['rand_mult'] = rng.randint(1, 4)
            folder_dict['active_cycle'] = True
            # Record pick times
            folder_dict['last_main_time']  = now
            folder_dict['last_float_time'] = now
        # Periodic active picks
        else:
            # Build jittered time thresholds (in seconds)
            jitter = rng.uniform(-rand_mult / LOCAL_CONTROLFPS, rand_mult / LOCAL_CONTROLFPS)
            min_interval = max(1.0, 1.0 + jitter)
            max_interval = max(min_interval, 20.0 + jitter)

            # MAIN trigger
            interval_main = (LOCAL_CONTROLFPS * (1 + (rand_mult + rand_mult // 2))) / LOCAL_CONTROLFPS  # seconds
            last_main = folder_dict['last_main_time']
            elapsed_main = now - last_main
            if elapsed_main >= min_interval and (math.isclose(( (index % (LOCAL_CONTROLFPS * (1 + (rand_mult + rand_mult // 2))))), 3 + rand_mult)):
                main_folder = folder_dict['pre_main'].popleft()
                #print("main folder is ", main_folder)
                folder_dict['rand_mult']       = rng.randint(1, 9)
                folder_dict['last_main_time']  = now
            elif elapsed_main >= max_interval:
                main_folder = folder_dict['pre_main'].popleft()
                folder_dict['last_main_time'] = now

            # FLOAT trigger
            interval_float = (((1 + rand_mult // 3) + LOCAL_CONTROLFPS) * rand_mult) / LOCAL_CONTROLFPS  # seconds
            last_float = folder_dict['last_float_time']
            elapsed_float = now - last_float
            if elapsed_float >= min_interval and (math.isclose((index % ((1 + rand_mult // 3 + LOCAL_CONTROLFPS) * rand_mult)), 7 + rand_mult)):
                float_folder = folder_dict['pre_float'].popleft()
                folder_dict['rand_mult']        = rng.randint(1, 12)
                folder_dict['last_float_time']  = now
            elif elapsed_float >= max_interval:
                float_folder = folder_dict['pre_float'].popleft()
                folder_dict['last_float_time'] = now

        # Refill pre-roll queues when low
        if len(folder_dict['pre_main']) < main_folder_count:
            seq = list(range(1, main_folder_count))
            rng.shuffle(seq)
            folder_dict['pre_main'].extend(seq)
        if len(folder_dict['pre_float']) < float_folder_count:
            seq = list(range(0, float_folder_count))
            rng.shuffle(seq)
            folder_dict['pre_float'].extend(seq)
    else:
        # MIDI-driven case (unchanged)
        note, channel, _ = control_data_dictionary['Note_On']
        modulation, _  = control_data_dictionary['Modulation']
        mod_value = int(modulation / 127 * float_folder_count)
        float_folder = mod_value % float_folder_count
        main_folder  = (note % 12) % main_folder_count

    # Save and return updated selection
    folder_dict['Main_and_Float_Folders'] = (main_folder, float_folder)
    return main_folder, float_folder