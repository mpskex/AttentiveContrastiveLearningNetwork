import time
import os
import psutil
import numpy as np

def BtoMB(bytes):
    return bytes / (1024 * 1024)

def get_start_time():
    return time.time()

def get_elapsed_since(start):
    return time.time() - start

def timetostr(start):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start))


def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def track(func):
    def wrapper(*args, **kwargs):
        mem_before = get_process_memory()
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = timetostr(get_elapsed_since(start))
        mem_after = get_process_memory()
        print("{}: memory before: {:,} MB, after: {:,} MB, consumed: {:,} MB; exec time: {}".format(
            func.__name__,
            BtoMB(mem_before), BtoMB(mem_after), BtoMB(mem_after) - BtoMB(mem_before),
            elapsed_time))
        return result
    return wrapper

def statistic_track(func, samples=100):
    def wrapper(*args, **kwargs):
        mem_used = []
        time_used = []
        mem_init = BtoMB(get_process_memory())
        result = None
        for n in range(samples):
            del result
            mem_before = BtoMB(get_process_memory())
            start = time.time()
            result = func(*args, **kwargs)
            elapsed_time = get_elapsed_since(start)
            mem_after = BtoMB(get_process_memory())
            time_used.append(elapsed_time)
            mem_used.append(mem_after - mem_before)
        mem_end = BtoMB(get_process_memory())
        mem_used, time_used = list(map(np.array, [mem_used, time_used]))
        print("{}: memory init: {:,} MB, memory end: {:,} MB, consumed avg: {:,} MB; exec time avg: {} ms".format(
            func.__name__,
            mem_init, mem_end, np.mean(mem_used), np.mean(time_used)*1000))
        return result
    return wrapper