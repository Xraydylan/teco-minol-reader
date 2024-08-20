import cv2
from cv2 import VideoCapture
import time
from collections import deque
from threading import Thread
import requests
from typing import List, Dict, cast
from ipaddress import IPv4Address
import queue

from .utils import KillThread, get_ip_oct

"""
What is needed:

Stream of single camera when needed!
-> Background updates not needed
-> Low update rate

-> Grayscale only cv2.COLOR_BGR2GRAY
"""


class ESP_CAM:
    def __init__(self, ip: IPv4Address, name: str, config: Dict, timing: Dict):
        self.ip = ip
        self.name = name
        self.timing = timing

        self.id = get_ip_oct(self.ip)

        self.base_url = f"http://{self.ip}"
        self.stream_url = f"{self.base_url}:81/stream"
        self.config_url = f"{self.base_url}/control"
        self.status_url = f"{self.base_url}/status"

        self.capture: VideoCapture = cast(VideoCapture, None)

        self.selected = False  # True if selected in GUI
        self.online = False  # True if ESP is online
        self.ready = False  # True if ESP ready to stream
        self.configured = False  # True if ESP fully configured
        self.connect_lock = False  # True if in connection process

        self.pre_connect_led_skip = True  # Skips led on when connecting
        self.pre_deselect_led_state = True  # Saves led state before deselect

        self.configuration = config
        self.default_config = self.configuration["default"]
        self.led_config = self.configuration["led_config"]

        self.led_intensity = self.led_config["default_intensity"]
        self.led_status = False

        self.delay = self.timing["ESP_CAM_delay"]
        self.timeout = self.timing["ESP_CAM_timeout"]

        self.deque: deque = deque(maxlen=1)
        self.thread: KillThread = cast(KillThread, None)

        self.request_queue = queue.Queue()

        self.request_thread: Thread = Thread(target=self._request_worker, daemon=True)
        self.request_thread.start()

    def start_worker(self):
        print(f"Start Worker: {self.base_url}")
        if self.thread is not None:
            self.thread.kill()

        self.thread = KillThread(target=self._fetch_frame,
                                 task_delay=self.delay,
                                 task_delay_check=self.timing.get("ESP_CAM_delay_check"),
                                 task_sleep_delay=self.timing.get("KillThread_sleep_delay"),
                                 task_sleep_delay_check=self.timing.get("KillThread_sleep_delay_check"))
        # self.thread.sleep()
        self.thread.start()

    def _fetch_frame(self):
        # if not self.selected:
        #    return

        # About if not ready
        if not self.ready:
            return

        # Abort if capture not initialized
        if self.capture is None:
            return

        try:
            # Check if capture is still opened
            if not self.capture.isOpened():
                return

            status, frame = self.capture.read()
            if status:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.deque.append(image)
            else:
                self.capture.release()
        except Exception as e:
            print(f"Caught error: {e}")

    def _request_worker(self):
        while 1:
            item = self.request_queue.get()  # get is blocking
            url, params = item
            try:
                r = requests.get(url=url, params=params, timeout=self.timeout)
                print(r)
            except:
                print(f"Request timout of {self.name}")

    def connect(self, wait=True, restart_worker=True):
        print(f"Connect: {self.ip}")
        if not self.online:
            return
        if self.thread is None or restart_worker:
            self.start_worker()
        self._connection_setup(wait=wait)

    def disconnect(self, kill_worker=True):
        print(f"Disconnected: {self.ip}")
        if self.thread is not None and kill_worker:
            self.thread.kill()
            self.thread = None

        if self.capture is not None:
            self.capture.release()

        self.ready = False

    def verify_network_stream(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            return False
        cap.release()
        return True

    def set_online_status(self, status, wait=True, affect_worker=True):
        # print(f"{status=}   {self.online=}")

        # If status is true and ESP already online -> do nothing
        if status and self.online:
            return

        # If status then ESP offline
        if status:
            self.online = True
            if not self.connect_lock:
                self.connect(wait=wait, restart_worker=affect_worker)
            if self.selected:
                self.thread.awake()
            else:
                self.led_off()
            return

        # If ESP is offline -> do nothing
        if not self.online:
            return

        # Turn off ESP
        self.online = False
        self.configured = False
        self.disconnect(kill_worker=affect_worker)

    def select(self):
        if self.selected:
            return

        self.selected = True
        if self.thread is None:
            return
        self.thread.awake()
        if self.pre_deselect_led_state:
            self.led_on()

    def deselect(self):
        if not self.selected:
            return

        self.selected = False
        if self.thread is None:
            return
        self.thread.sleep()
        self.pre_deselect_led_state = self.led_status
        self.led_off()

    def _connection_setup(self, wait=True):
        self.connect_lock = True

        def load_network_stream_thread():
            capture = cv2.VideoCapture(self.stream_url)
            if capture.isOpened():
                self.capture = capture
                self.ready = True
                self.check_config()
            else:
                capture.release()

        load_stream_thread = Thread(target=load_network_stream_thread)
        load_stream_thread.daemon = True
        load_stream_thread.start()

        if wait:
            load_stream_thread.join()
        self.connect_lock = False

    def check_config(self):
        data = {}
        try:
            # Request here is ok because it is in side thread and not blocking for main program
            r = requests.get(url=self.status_url, timeout=self.timeout)
            data = r.json()
        except Exception as e:
            print(e)
            self.configured = False
            return

        for key, value in self.default_config.items():
            if key in data:

                # make sure LED is off
                if key == "led_intensity":
                    self.led_off()
                    continue

                if data[key] != value:
                    self._config(key, value)

        # LED config is done in select/deselect
        # if not self.configured:
        #     self._led_send_config()

        self.configured = True

    # CONFIG request error
    def _config(self, var, val):
        params = {
            "var": var,
            "val": val
        }
        self.request_queue.put([self.config_url, params])

    def reconfigure(self):
        for key, value in self.default_config.items():
            self._config(key, value)

        self.led_set_default()

    def led_off(self):
        self._led_send_config(intensity=0)

    def led_on(self):
        self._led_send_config()

    def led_set_intensity(self, intensity_percent):
        if intensity_percent > 1:
            intensity_percent = 1
        self.led_intensity = self.led_config["max_intensity"] * intensity_percent
        self._led_send_config()

    def led_set_default(self):
        self.led_intensity = self.led_config["default_intensity"]
        self._led_send_config()

    def _led_send_config(self, intensity=None):
        if intensity is None:
            intensity = self.led_intensity

        if self.pre_connect_led_skip:
            self.pre_connect_led_skip = False
            intensity = 0

        if intensity > 0:
            self.led_status = True
        else:
            self.led_status = False
        self._config(var="led_intensity", val=intensity)

    def get_frame(self):
        if len(self.deque):
            return self.online, self.deque[-1]
        return False, None

    def get_info(self):
        return self.name, self.id

    def get_online(self):
        return self.online

    def reset(self):
        self.ready = False
        self.selected = False
        self.disconnect()

        time.sleep(1)
        self.selected = True
        self.connect(restart_worker=True)

    def get_led_status(self):
        if not self.online:
            return 2
        return int(self.led_status)
