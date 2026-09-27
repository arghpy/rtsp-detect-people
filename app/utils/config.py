import sys
import app.utils.files
import app.utils.logger

CONFIG = {}
CONFIG["CONFIDENCE_MIN"] = 0.55
CONFIG["TIMEOUT"] = 60
CONFIG["HA_ENTITY_ID"] = None
CONFIG["HA_ENTITY_TYPE"] = None
CONFIG["HA_HEADERS"] = None
CONFIG["HA_TOKEN"] = None
CONFIG["HA_URL"] = None
CONFIG["YOLO_MODEL"] = "yolo11m.pt"
CONFIG["YOLO_BATCH"] = 8
CONFIG["YOLO_IMGSZ"] = 640
CONFIG["VIDEO_NAME"] = None
CONFIG["VIDEO_PATH"] = None


def process_configuration(config_file):
    global CONFIG

    configuration = app.utils.files.load_json_file(config_file)

    # General
    try:
        CONFIG["TIMEOUT"] = int(configuration.get("timeout"))  # Secs
        CONFIG["CONFIDENCE_MIN"] = float(configuration.get("confidence"))
    except KeyError:
        app.utils.logger.eprint("Default values will be used")

    # YOLO
    try:
        CONFIG["YOLO_MODEL"] = configuration["yolo"]["model"]
        CONFIG["YOLO_BATCH"] = int(configuration["yolo"]["batch_size"])
        CONFIG["YOLO_IMGSZ"] = int(configuration["yolo"]["imgsz"])
    except KeyError:
        app.utils.logger.eprint("Default values will be used")

    # RTSP
    try:
        CONFIG["VIDEO_NAME"] = configuration["rtsp"]["save_video"]["name"]
        CONFIG["VIDEO_PATH"] = configuration["rtsp"]["save_video"]["path"]
    except KeyError:
        app.utils.logger.eprint("Video won't pe saved")

    # Home Assistant
    try:
        HA_TOKEN = configuration["home-assistant"]["token"]
        HA_URL = configuration["home-assistant"]["base_http_url"]
        CONFIG["HA_ENTITY_ID"] = configuration["home-assistant"]["entity"]["id"]
        CONFIG["HA_ENTITY_TYPE"] = configuration["home-assistant"]["entity"]["type"]
        CONFIG["HA_URL"] = f"{HA_URL}/api/services/{CONFIG['HA_ENTITY_TYPE']}"
        CONFIG["HA_HEADERS"] = {
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json",
        }
    except KeyError:
        app.utils.logger.eprint("home assistant won't be notified")
