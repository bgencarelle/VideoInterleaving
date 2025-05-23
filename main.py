import image_display
import make_file_lists
from  settings import CLOCK_MODE

def main(clock=CLOCK_MODE):
    make_file_lists.process_files()
    image_display.run_display(clock)


if __name__ == "__main__":
    main()
