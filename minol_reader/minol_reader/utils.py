import os
import time
from typing import Dict, cast
from ipaddress import IPv4Address
from threading import Thread
from pythonping import ping
from time import perf_counter

from typing import List


def get_ip(subnetmask, ip_oct):
    ip_string = str(subnetmask)
    return IPv4Address('.'.join(ip_string.split('.')[:-1] + [str(ip_oct)]))


def get_ip_oct(ip):
    ip_string = str(ip)
    return ip_string.split('.')[-1]


def get_base_path(file, levels=2):
    path = os.path.abspath(__file__)
    for l in range(levels):
        path = os.path.dirname(path)
    return path + "/"


class Task:
    def __init__(self, name, func, delay, active=None):
        if active is None:
            active = True

        self.name = name
        self.func = func
        self.delay = delay
        self.active = active
        self.last = 0

    def check_activate(self, t=None):
        if not self.active:
            return

        if t is None:
            t = perf_counter()  # perf_counter() or time.time(); perf_counter() is better

        if t - self.last > self.delay:
            self.func()
            self.last = t

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False


"""
KillThread has 4 delays/time intervals:
task_delay: Delay between task executions
task_delay_check: Delay between checks when active

task_sleep_delay: Delay between task executions when sleeping
task_sleep_delay_check: Delay between checks when sleeping
"""


class KillThread:
    def __init__(self, target, task_delay, task_delay_check=None, task_sleep_delay=None, task_sleep_delay_check=None,
                 daemon=None):
        if daemon is None:
            daemon = True
        if task_delay_check is None:
            task_delay_check = task_delay * 0.1
        if task_sleep_delay is None:
            task_sleep_delay = 5
        if task_sleep_delay_check is None:
            task_sleep_delay_check = task_sleep_delay * 0.2

        self.target = target
        self.task_delay = task_delay
        self.task_delay_check = task_delay_check

        self.task_sleep_delay = task_sleep_delay
        self.task_sleep_delay_check = task_sleep_delay_check

        self.thread: Thread = Thread(target=self._worker, daemon=True)
        self.thread.daemon = daemon

        self.task: Task = Task(f"Kill_Task_{self.thread.name}", self.target, self.task_delay)

        self.sleeping = False
        self.running = 0

    def start(self):
        if self.running:
            return
        self.running = 1
        self.thread.start()

    def sleep(self):
        self.sleeping = True
        self.task.delay = self.task_sleep_delay

    def awake(self):
        self.sleeping = False
        self.task.activate()
        self.task.delay = self.task_delay

    def deactivate(self):
        self.task.deactivate()

    def kill(self):
        print(f"Init Kill: {self.thread.name}")
        self.task.deactivate()
        self.running = 0

    def _worker(self):
        while self.running:
            self.task.check_activate()

            if self.sleeping:
                time.sleep(self.task_sleep_delay_check)
                continue
            time.sleep(self.task_sleep_delay_check)
        print(f"Killed thread: {self.thread.name}")


class Ping:
    def __init__(self, ip_list):
        self.ips: List[str] = ip_list

        self.max_index = len(ip_list)
        self.index = 0

        self.pings = [False] * self.max_index

    def ping_next(self):
        self.index += 1
        if self.index >= self.max_index:
            self.index = 0

        ip = self.ips[self.index]
        if self._check_ping(ip):
            self.pings[self.index] = True
            return [self.index, True]

        if self.pings[self.index]:
            if self._check_ping(ip):
                self.pings[self.index] = True
                return [self.index, True]

        self.pings[self.index] = False
        return [self.index, False]

    def ping_all(self):
        for index, ip in enumerate(self.ips):
            if self._check_ping(ip):
                self.pings[index] = True
                continue

            if self.pings[index]:
                if self._check_ping(ip):
                    self.pings[index] = True
                    continue
            self.pings[index] = False

    def _check_ping(self, ip):
        return ping(str(ip), count=2, timeout=0.25).success()

    def set_offline(self, cam_num):
        self.pings[cam_num] = False

    def get_status_cam_num(self):
        return [(index, online) for index, online in enumerate(self.pings)]
