import certifi
import requests

import app.utils.logger


def send_ntfy(base_url, tag, title, body, attachment_path, attachment_name):
    app.utils.logger.pprint("Person detected. Sending notification")
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
            verify=certifi.where()
        )
    r.raise_for_status()
