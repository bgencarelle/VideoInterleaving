import math
import random
import time
from collections import deque
from globals import control_data_dictionary, folder_dictionary
from settings import FPS, CLOCK_MODE
from utilities.csv_list_maker import main_folder_path

def update_folder_selection(index, float_folder_count, main_folder_count):
    """
    Maintains persistent random timing for folder switching.
    Folders reset to 0 in rest zones and switch based on randomized modulus logic.
    Uses a single RNG and pre-rolled sequences for improved entropy,
    while preserving the rest → first → periodic structure.
    Enforces a minimum ~1 s gap, caps at ~20 s, and adds slight random jitter.
    Timing is based on real elapsed time, not ping-pong index.
    """
    # Retrieve last folders
    main_folder, float_folder = folder_dictionary.get(
        'Main_and_Float_Folders', (0, 0)
    )

    # SINGLE RNG instance
    if 'rng' not in folder_dictionary:
        folder_dictionary['rng'] = random.Random()
        folder_dictionary['rng'].seed(time.time())
    rng = folder_dictionary['rng']

    # Initialize last-pick times for spacing enforcement
    now = time.time()
    if 'last_main_time' not in folder_dictionary:
        folder_dictionary['last_main_time'] = now
    if 'last_float_time' not in folder_dictionary:
        folder_dictionary['last_float_time'] = now

    # PRE-ROLL sequences for main and float (skip folder 0 for "boring" main)
    if 'pre_main' not in folder_dictionary:
        seq = list(range(1, main_folder_count))
        folder_dictionary['pre_main'] = deque()
        for _ in range(2):
            rng.shuffle(seq)
            folder_dictionary['pre_main'].extend(seq)
    if 'pre_float' not in folder_dictionary:
        seq = list(range(0, float_folder_count))
        folder_dictionary['pre_float'] = deque()
        for _ in range(2):
            rng.shuffle(seq)
            folder_dictionary['pre_float'].extend(seq)

    # Timing parameters via single RNG
    if 'rand_mult' not in folder_dictionary:
        folder_dictionary['rand_mult'] = rng.randint(1, 9)
    if 'rand_start' not in folder_dictionary:
        # rest duration in frames
        folder_dictionary['rand_start'] = rng.randint(FPS, int(3.5 * FPS))

    rand_mult = folder_dictionary['rand_mult']
    rand_start = folder_dictionary['rand_start']

    # Active-cycle flag
    if 'active_cycle' not in folder_dictionary:
        folder_dictionary['active_cycle'] = False

    if CLOCK_MODE == 255:
        # Rest zone: both folders = 0
        if index <= rand_start:
            if folder_dictionary['active_cycle']:
                # On entering rest, reroll next start
                folder_dictionary['rand_start'] = rng.randint(FPS, int(3.5 * FPS))
            folder_dictionary['active_cycle'] = False
            main_folder = 0
            float_folder = 0
        # First active pick
        elif not folder_dictionary['active_cycle']:
            # Pop initial values
            float_folder = folder_dictionary['pre_float'].popleft()
            main_folder  = folder_dictionary['pre_main'].popleft()
            # New tempo for next beats
            folder_dictionary['rand_mult'] = rng.randint(1, 4)
            folder_dictionary['active_cycle'] = True
            # Record pick times
            folder_dictionary['last_main_time']  = now
            folder_dictionary['last_float_time'] = now
        # Periodic active picks
        else:
            # Build jittered time thresholds (in seconds)
            jitter = rng.uniform(-rand_mult / FPS, rand_mult / FPS)
            min_interval = max(1.0, 1.0 + jitter)
            max_interval = max(min_interval, 20.0 + jitter)

            # MAIN trigger
            interval_main = (FPS * (1 + (rand_mult + rand_mult // 2))) / FPS  # seconds
            last_main = folder_dictionary['last_main_time']
            elapsed_main = now - last_main
            if elapsed_main >= min_interval and (math.isclose(( (index % (FPS * (1 + (rand_mult + rand_mult // 2))))), 3 + rand_mult) ):
                main_folder = folder_dictionary['pre_main'].popleft()
                folder_dictionary['rand_mult']       = rng.randint(1, 9)
                folder_dictionary['last_main_time']  = now
            elif elapsed_main >= max_interval:
                main_folder = folder_dictionary['pre_main'].popleft()
                folder_dictionary['last_main_time'] = now

            # FLOAT trigger
            interval_float = (((1 + rand_mult // 3) + FPS) * rand_mult) / FPS  # seconds
            last_float = folder_dictionary['last_float_time']
            elapsed_float = now - last_float
            if elapsed_float >= min_interval and (math.isclose((index % ((1 + rand_mult // 3 + FPS) * rand_mult)), 7 + rand_mult)):
                float_folder = folder_dictionary['pre_float'].popleft()
                folder_dictionary['rand_mult']        = rng.randint(1, 12)
                folder_dictionary['last_float_time']  = now
            elif elapsed_float >= max_interval:
                float_folder = folder_dictionary['pre_float'].popleft()
                folder_dictionary['last_float_time'] = now

        # Refill pre-roll queues when low
        if len(folder_dictionary['pre_main']) < main_folder_count:
            seq = list(range(1, main_folder_count))
            rng.shuffle(seq)
            folder_dictionary['pre_main'].extend(seq)
        if len(folder_dictionary['pre_float']) < float_folder_count:
            seq = list(range(0, float_folder_count))
            rng.shuffle(seq)
            folder_dictionary['pre_float'].extend(seq)
    else:
        # MIDI-driven case (unchanged)
        note, channel, _ = control_data_dictionary['Note_On']
        modulation, _  = control_data_dictionary['Modulation']
        mod_value = int(modulation / 127 * float_folder_count)
        float_folder = mod_value % float_folder_count
        main_folder  = (note % 12) % main_folder_count

    # Save and return updated selection
    folder_dictionary['Main_and_Float_Folders'] = (main_folder, float_folder)
    return main_folder, float_folder