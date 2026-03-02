import requests

HA_LIGHT = False


def ha_trigger_boolean(url, headers, entity_id, request: bool):
    state = "turn_on" if request else "turn_off"
    url = f"{url}/{state}"
    payload = {
        "entity_id": f"{entity_id}"
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
