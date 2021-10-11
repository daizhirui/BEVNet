import time
from collections import defaultdict
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self._t0 = None
        self._t1 = None
        self._elapsed = 0
        self._t_lap = 0
        self._laps = []
        self._running = False

    @property
    def t0(self):
        return self._t0

    @property
    def t1(self):
        return self._t1

    @property
    def elapsed(self):
        if self._running:
            return time.time() - self._t0 + self._elapsed
        else:
            return self._elapsed

    @property
    def laps(self):
        return self._laps.copy()

    def start(self):
        if self._running:
            return
        self._running = True
        self._t0 = time.time()

    def stop(self):
        if not self._running:
            return
        self._t1 = time.time()
        self._elapsed += self._t1 - self._t0
        self._running = False

    def lap(self):
        if not self._running:
            return
        new_t_lap = self.elapsed
        self._laps.append(new_t_lap - self._t_lap)
        self._t_lap = new_t_lap


class TimerManager:
    def __init__(self):
        self.timers = defaultdict(Timer)

    @contextmanager
    def timing(self, name=None):
        if name is None:
            timer = Timer()
        else:
            timer = self.timers[name]
        try:
            timer.start()
            yield timer
        finally:
            timer.stop()
