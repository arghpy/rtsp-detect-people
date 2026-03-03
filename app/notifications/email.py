from core import pprint
import time
import smtplib
from email.message import EmailMessage

SEND_EMAIL = False


def send_email_report(save_image_path, save_image_type, config):
    """Send email based on the environment variables"""
    pprint("Person detected. Sending email")

    # Create the container email message.
    msg = EmailMessage()
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S")
    msg["Subject"] = config["email"]["subject"] + f": {current_time}"
    msg["From"] = config["email"]["user"]
    msg["To"] = ", ".join(config["email"]["recipients"])

    # Open the image in binary mode
    with open(save_image_path, "rb") as fp:
        img_data = fp.read()
        msg.add_attachment(
            img_data,
            maintype="image",
            subtype=save_image_type,
            filename=save_image_path,
        )

    with smtplib.SMTP_SSL(config["email"]["server"], config["email"]["port"]) as s:
        s.login(config["email"]["user"], config["email"]["password"])
        s.send_message(msg)
