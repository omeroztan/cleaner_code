# todo convert it to seconds

import time

class Stopwatch:
    def __init__(self):
        self.start = 0
        self.stop = 0
        self.value = self.stop - self.start

    def start_method(self):
        self.start = time.time()

    def stop_method(self):
        self.stop = time.time()