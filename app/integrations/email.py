import smtplib
import time
from email.message import EmailMessage

from app.utils.config import CONFIG
from app.utils.logger import pprint


def send_email_report(save_image_path):
    """Send email based on the environment variables"""
    pprint("Person detected. Sending email")

    # Create the container email message.
    msg = EmailMessage()
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S")
    msg["Subject"] = CONFIG["EMAIL_SUBJECT"] + f": {current_time}"
    msg["From"] = CONFIG["EMAIL_FROM"]
    msg["To"] = ", ".join(CONFIG["EMAIL_TO"])

    # Open the image in binary mode
    with open(save_image_path, "rb") as fp:
        img_data = fp.read()
        msg.add_attachment(
            img_data,
            maintype="image",
            subtype="jpeg",
            filename=save_image_path,
        )

    with smtplib.SMTP_SSL(CONFIG["EMAIL_SERVER"], CONFIG["EMAIL_PORT"]) as s:
        s.login(CONFIG["EMAIL_FROM"], CONFIG["EMAIL_PASSWORD"])
        s.send_message(msg)
