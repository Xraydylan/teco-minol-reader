from minol_reader import app

config_path = "config.json"

def main():
    config = app.MinolApp.read_config(config_path)
    version = config.get("version")

    print("Starting Minol Reader App")
    print(f"Version: {version}")

    esp_ip = config.get("esp_ip")
    delay = config.get("delay")
    save_images = config.get("save_images")
    monotone_detection = config.get("monotone_detection")
    mqtt_settings = config.get("mqtt_settings")
    exit_on_error = config.get("exit_on_error")

    image_parameters = config.get("image_parameters")

    delay = delay * 60  # convert to minutes!
    mqtt_settings["reconnect_wait"] = mqtt_settings.get("reconnect_wait", 5) * 60  # convert to minutes!

    m_app = app.MinolApp(str(esp_ip),
                         delay,
                         mqtt_settings,
                         save_images=save_images,
                         monotone_detection=monotone_detection,
                         exit_on_error=exit_on_error,
                         image_parameters=image_parameters)
    m_app.start()


if __name__ == '__main__':
    main()
