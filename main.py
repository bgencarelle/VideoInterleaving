import sys, threading

# Tee stdout/stderr to a line-buffered runtime.log
class Tee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.stream.write(data)
            self.stream.flush()
            self.log_file.write(data)
            self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

# open in line-buffered mode
log_file = open("runtime.log", "w", buffering=1)
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

import image_display  # now all prints get captured immediately
import make_file_lists
from settings import CLOCK_MODE

def main(clock=CLOCK_MODE):
    make_file_lists.process_files()
    image_display.run_display(clock)

if __name__ == "__main__":
    main()
