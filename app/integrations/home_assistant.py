import ssl

import certifi
import requests

import app.utils.config
import app.utils.logger

ssl_context = ssl.create_default_context(cafile=certifi.where())


def ha_trigger_boolean(request: bool):
    state = "turn_on" if request else "turn_off"
    url = f"{app.utils.config.CONFIG['HA_URL']}/{state}"
    payload = {"entity_id": f"{app.utils.config.CONFIG['HA_ENTITY_ID']}"}

    try:
        response = requests.post(
            url,
            headers=app.utils.config.CONFIG["HA_HEADERS"],
            json=payload,
            verify=certifi.where(),
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        app.utils.logger.eprint(f"HA trigger failed for {url}: {e}")
