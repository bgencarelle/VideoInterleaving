import os
import get_folders_list
import calculators
import image_display
import make_file_lists

MTC_CLOCK = 0
MIDI_CLOCK = 1
MIXED_CLOCK = 2
CLIENT_MODE = 3
FREE_CLOCK = 255

CLOCK_MODE = FREE_CLOCK

def main(clock=CLOCK_MODE):
    image_display.display_and_run(clock)


if __name__ == "__main__":
    main()
