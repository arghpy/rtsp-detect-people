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
CONFIG["NTFY_TAG"] = None
CONFIG["NTFY_URL"] = None
CONFIG["RTSP_FEED"] = None
CONFIG["RTSP_URL"] = None


def process_configuration(config_file):
    global CONFIG

    configuration = app.utils.files.load_json_file(config_file)

    try:
        # RTSP
        RTSP_USER = configuration["rtsp"]["user"]
        RTSP_PASSWORD = configuration["rtsp"]["password"]
        CONFIG["RTSP_FEED"] = configuration["rtsp"]["feed"]
        CONFIG["RTSP_URL"] = f"rtsp://{RTSP_USER}:{RTSP_PASSWORD}@{CONFIG['RTSP_FEED']}"
    except KeyError as e:
        app.utils.logger.eprint(f"[CONFIG] Mandatory config option missing: {e}")
        sys.exit(1)

    try:
        # General
        CONFIG["TIMEOUT"] = int(configuration.get("timeout"))  # Secs
        CONFIG["CONFIDENCE_MIN"] = float(configuration.get("confidence"))
    except KeyError:
        app.utils.logger.eprint("[CONFIG] Default values will be used")

    try:
        # YOLO
        CONFIG["YOLO_MODEL"] = configuration["yolo"]["model"]
        CONFIG["YOLO_BATCH"] = int(configuration["yolo"]["batch_size"])
        CONFIG["YOLO_IMGSZ"] = int(configuration["yolo"]["imgsz"])
    except KeyError:
        app.utils.logger.eprint("[CONFIG] Default values will be used")

    try:
        # NTFY
        CONFIG["NTFY_URL"] = configuration["ntfy"]["url"]
        CONFIG["NTFY_TAG"] = configuration["ntfy"]["tag"]
    except KeyError:
        app.utils.logger.eprint("[CONFIG] ntfy won't be sent")

    try:
        # Home Assistant
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
        app.utils.logger.eprint("[CONFIG] home assistant won't be notified")
