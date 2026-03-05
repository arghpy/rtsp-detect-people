import requests
from app.utils.logger import pprint


def send_ntfy(base_url, tag, title, body, attachment_path, attachment_name):
    pprint("Person detected. Sending notification")
    url = f"{base_url}/{tag}"
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
