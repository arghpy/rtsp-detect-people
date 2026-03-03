import requests
from core import pprint
from config import CONFIG

SEND_NTFY = False


def send_ntfy(title, body, attachment_path, attachment_name):
    pprint("Person detected. Sending notification")
    url = f"{CONFIG.NTFY_URL}/{CONFIG.NTFY_TAG}"
    with open(attachment_path, "rb") as f:
        r = requests.post(
            url,
            data=f,
            headers={
                "Filename": attachment_name,
                "Title": title,
                "Message": body,
            },
        )
    r.raise_for_status()
