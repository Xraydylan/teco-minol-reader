from minol_reader import app

config_path = "config.json"

def main():
    config = app.MinolApp.read_config(config_path)

    esp_ip = config.get("esp_ip")
    delay = config.get("delay")
    save_images = config.get("save_images")
    mqtt_settings = config.get("mqtt_settings")
    image_parameters = config.get("image_parameters")

    delay = delay * 60  # convert to seconds
    mqtt_settings["reconnect_wait"] = mqtt_settings.get("reconnect_wait", 5) * 60  # convert to seconds

    m_app = app.MinolApp(str(esp_ip), delay, mqtt_settings, image_parameters=image_parameters)

    m_app.calibration()


if __name__ == '__main__':
    main()
