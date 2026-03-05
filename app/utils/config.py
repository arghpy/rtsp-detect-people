from app.utils.files import load_json_file
from app.utils.logger import eprint

CONFIG = {}
CONFIG["CONFIDENCE_MIN"] = None
CONFIG["HA_ENTITY_ID"] = None
CONFIG["HA_ENTITY_TYPE"] = None
CONFIG["HA_HEADERS"] = None
CONFIG["HA_TOKEN"] = None
CONFIG["HA_URL"] = None
CONFIG["MODEL"] = None
CONFIG["NTFY_TAG"] = None
CONFIG["NTFY_URL"] = None
CONFIG["RTSP_FEED"] = None
CONFIG["RTSP_URL"] = None
CONFIG["TIMEOUT"] = None
CONFIG["VIDEO_FPS"] = None
CONFIG["VIDEO_FPS"] = None
CONFIG["VIDEO_NAME"] = None
CONFIG["VIDEO_PATH"] = None


def process_configuration(config_file):
    global CONFIG

    configuration = load_json_file(config_file)

    try:
        # General
        CONFIG["TIMEOUT"] = int(configuration["timeout"])  # Secs
        CONFIG["MODEL"] = configuration["model"]  # YOLO Model to use
        CONFIG["CONFIDENCE_MIN"] = float(configuration["confidence"])

        # RTSP
        RTSP_USER = configuration["rtsp"]["user"]
        RTSP_PASSWORD = configuration["rtsp"]["password"]
        CONFIG["RTSP_FEED"] = configuration["rtsp"]["feed"]
        CONFIG["RTSP_URL"] = f"rtsp://{RTSP_USER}:{RTSP_PASSWORD}@{CONFIG['RTSP_FEED']}"
    except KeyError as e:
        eprint(f"[CONFIG] Mandatory config option missing: {e}")

    try:
        CONFIG["VIDEO_NAME"] = configuration["rtsp"]["save_video"]["name"]
        CONFIG["VIDEO_PATH"] = configuration["rtsp"]["save_video"]["path"]
        CONFIG["VIDEO_FPS"] = int(configuration["rtsp"]["save_video"]["optional_force_fps"])
    except KeyError:
        eprint("[CONFIG] Video won't pe saved")

    try:
        # Email
        CONFIG["NTFY_URL"] = configuration["ntfy"]["url"]
        CONFIG["NTFY_TAG"] = configuration["ntfy"]["tag"]
        CONFIG["EMAIL_SUBJECT"] = configuration["email"]["subject"]
        CONFIG["EMAIL_FROM"] = configuration["email"]["user"]
        CONFIG["EMAIL_TO"] = configuration["email"]["recipients"]
        CONFIG["EMAIL_SERVER"] = configuration["email"]["server"]
        CONFIG["EMAIL_PORT"] = configuration["email"]["port"]
        CONFIG["EMAIL_PASSWORD"] = configuration["email"]["password"]
    except KeyError:
        eprint("[CONFIG] email won't be sent")

    try:
        # NTFY
        CONFIG["NTFY_URL"] = configuration["ntfy"]["url"]
        CONFIG["NTFY_TAG"] = configuration["ntfy"]["tag"]
    except KeyError:
        eprint("[CONFIG] ntfy won't be sent")

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
        eprint("[CONFIG] home assistant won't be notified")
