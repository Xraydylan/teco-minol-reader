from .minol_reader.minol import MINOL
from .mqtt.mqtt_manager import MQTT_Manager

from pathlib import Path
import json

import time
from ipaddress import IPv4Address

# Type hinting
from typing import Dict
from numbers import Number


class MinolApp:
    def __init__(self, esp_ip: str, delay: Number, mqtt_settings: Dict[str, str], save_images: bool = None,
                 monotone_detection: bool = None, exit_on_error: bool = None, image_parameters: dict = None):
        if save_images is None:
            save_images = False
        if exit_on_error is None:
            exit_on_error = False
        self.esp_ip: str = esp_ip
        self.delay: float = float(delay)  # Delay between values in seconds
        self.mqtt_settings: Dict[str, str] = mqtt_settings

        self.minol: MINOL = MINOL(self.esp_ip, save_images=save_images, monotone_detection=monotone_detection,
                                  exit_on_error=exit_on_error, image_parameters=image_parameters)

        self.mqtt: MQTT_Manager = MQTT_Manager(
            client_name="Minol",
            broker=self.mqtt_settings["broker"],
            username=self.mqtt_settings["username"],
            password=self.mqtt_settings["password"],
            prefix=self.mqtt_settings.get("prefix"),
            topic_base=self.mqtt_settings.get("topic_base"),
            reconnect_wait=self.mqtt_settings.get("reconnect_wait")
        )

    def start(self):
        print("Starting MinolApp...")
        self.mqtt.start()
        self.minol.start()
        time.sleep(1)
        print("MinolApp Online")

        self._loop()

    def calibration(self):
        print("Starting Camera...")
        self.minol.start()
        print("Camera Online")
        self.minol.show_frame()

    def test_reading(self):
        print("Starting Camera...")
        self.minol.start()
        print("Camera Online")

        while 1:
            number = self.minol.get_number()
            print("=================================>  ", number, "\n")
            time.sleep(60)

    @staticmethod
    def read_config(config_file: str | Path):
        """
        Read configuration json file.
        Return the dictionary with the configuration.

        :param config_file: Path to the configuration file.
        :return: Dictionary with the configuration.
        """
        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as file:
            config = json.load(file)

        return config

    def _loop(self):
        while 1:
            print("Time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            value = self.minol.get_number()
            print(f"Value: {value}")

            if value is not None:
                self.mqtt.publish("value", str(value), 1, True)
                self.mqtt.publish("status", "OK", 1, True)
            else:
                self.mqtt.publish("status", "ValueError", 1, True)
            time.sleep(self.delay)
