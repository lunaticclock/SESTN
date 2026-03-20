import datetime
import time


class Timer:
    def __init__(self):
        self._begin_time = None
        self.allTime = 0
        self.frequency = 0

    def start(self):
        self._begin_time = time.perf_counter()

    def stop(self):
        interval = time.perf_counter() - self._begin_time
        self.allTime += interval
        self.frequency += 1
        return str(datetime.timedelta(seconds=interval))

    def sum(self):
        return self.allTime

    def GetTime(self):
        return str(datetime.timedelta(seconds=self.allTime))

    def avg(self):
        return self.allTime / self.frequency

    def __str__(self):
        return self.GetTime()
