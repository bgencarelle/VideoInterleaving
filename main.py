import sys, threading  # [Add: initialize logging before other imports]
class Tee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self.lock = threading.Lock()
    def write(self, data):
        with self.lock:
            self.stream.write(data)
            self.log_file.write(data)
    def flush(self):
        self.stream.flush()
        self.log_file.flush()

log_file = open("runtime.log", "a")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

import image_display  # [Existing imports now follow logging setup]
import make_file_lists
from settings import CLOCK_MODE

def main(clock=CLOCK_MODE):
    make_file_lists.process_files()
    image_display.run_display(clock)

if __name__ == "__main__":
    main()
