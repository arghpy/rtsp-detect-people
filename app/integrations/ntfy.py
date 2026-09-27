import certifi
import cv2
import numpy as np
import requests
import app.utils.logger


def compress_for_ntfy(jpeg_bytes, quality=80):
    """
    Authored by Claude.
    Re-compresses a JPEG at a lower quality to shrink file size,
    without changing its dimensions. Returns JPEG bytes, or the
    original bytes if something goes wrong.
    """
    img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes  # couldn't decode, don't touch it

    success, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes() if success else jpeg_bytes


def send_ntfy(base_url, tag, title, body, attachment_path, attachment_name):
    app.utils.logger.iprint("Person detected. Sending notification")
    url = f"{base_url}/{tag}"
    with open(attachment_path, "rb") as f:
        data = f.read()
    r = requests.post(
        url,
        timeout=300,
        data=data,
        headers={
            "Filename": attachment_name,
            "Title": title,
            "Message": body,
            "Content-Type": "image/jpeg",
        },
        verify=certifi.where()
    )
    r.raise_for_status()
