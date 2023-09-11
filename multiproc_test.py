import os
from threading import Thread
from threading import Event
from multiprocessing import Lock
import time
from multiprocessing import Process


def process_i(id, lock, shared_var):
    for i in range(5):
        with lock:
            shared_var += 1
            print("[Proces ", id, "] shared_var = ", shared_var)
        time.sleep(1)


if __name__ == "__main__":
    # create the shared lock
    lock = Lock()
    shared = 0
    # create a number of processes with different sleep times
    processes = [Process(target=process_i, args=(i, lock, shared)) for i in range(10)]
    # start the processes
    for process in processes:
        process.start()
    # wait for all processes to finish
    for process in processes:
        process.join()
