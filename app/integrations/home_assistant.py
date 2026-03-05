import requests
from app.utils.config import CONFIG


def ha_trigger_boolean(request: bool):
    state = "turn_on" if request else "turn_off"
    url = f"{CONFIG['HA_URL']}/{state}"
    payload = {"entity_id": f"{CONFIG['HA_ENTITY_ID']}"}

    response = requests.post(url, headers=CONFIG["HA_HEADERS"], json=payload)
    response.raise_for_status()
