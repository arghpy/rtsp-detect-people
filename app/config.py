from core import load_json_file, eprint

CONFIG = {}
CONFIG["CONFIDENCE_MIN"] = None
CONFIG["TIMEOUT"] = None
CONFIG["RTSP_FEED"] = None
CONFIG["RTSP_URL"] = None
CONFIG["NTFY_URL"] = None
CONFIG["NTFY_TAG"] = None
CONFIG["VIDEO_FPS"] = None
CONFIG["HA_TOKEN"] = None
CONFIG["HA_URL"] = None
CONFIG["HA_ENTITY_TYPE"] = None
CONFIG["HA_ENTITY_ID"] = None
CONFIG["HA_HEADERS"] = None
CONFIG["VIDEO_NAME"] = None
CONFIG["VIDEO_PATH"] = None


def process_configuration(config_file):
    global CONFIG

    configuration = load_json_file(config_file)

    try:
        # General
        CONFIG["TIMEOUT"] = int(configuration["timeout"])  # Secs
        CONFIG["CONFIDENCE_MIN"] = float(configuration["confidence"])

        # RTSP
        RTSP_USER = configuration["rtsp"]["user"]
        RTSP_PASSWORD = configuration["rtsp"]["password"]
        CONFIG["RTSP_FEED"] = configuration["rtsp"]["feed"]
        CONFIG["RTSP_URL"] = f"rtsp://{RTSP_USER}:{RTSP_PASSWORD}@{CONFIG["RTSP_FEED"]}"
    except KeyError as e:
        eprint(f"[CONFIG] Mandatory config option missing: {e}")

    try:
        CONFIG["VIDEO_NAME"] = configuration["rtsp"]["save_video"]["name"]
        CONFIG["VIDEO_PATH"] = configuration["rtsp"]["save_video"]["path"]
    except KeyError:
        eprint("[CONFIG] Video won't pe saved")

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
        CONFIG["HA_URL"] = f"{HA_URL}/api/services/{CONFIG["HA_ENTITY_TYPE"]}"
        CONFIG["HA_HEADERS"] = {
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json",
        }
    except KeyError:
        eprint("[CONFIG] home assistant won't be notified")
