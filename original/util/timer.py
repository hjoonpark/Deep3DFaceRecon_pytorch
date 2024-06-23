import os
import datetime
import time

class Timer():
    def __init__(self):
        self.time0 = time.time()

    def format_seconds(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return h, m, s

    def tic(self):
        self.time0 = time.time()

    def toc(self):
        now = datetime.datetime.now()
        dt = time.time()-self.time0
        h, m, s = self.format_seconds(dt)
        dt_str = "{:02.0f}:{:02.0f}:{:02.0f}".format(h, m, s)
        now_str = "{}-{:02}-{:02} {:02}:{:02}:{:02}s".format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        return now_str, dt_str