import smtplib
import time
from email.message import EmailMessage

import app.utils.config
import app.utils.logger


def send_email_report(save_image_path):
    """Send email based on the environment variables"""
    app.utils.logger.pprint("Person detected. Sending email")

    # Create the container email message.
    msg = EmailMessage()
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S")
    msg["Subject"] = app.utils.config.CONFIG["EMAIL_SUBJECT"] + f": {current_time}"
    msg["From"] = app.utils.config.CONFIG["EMAIL_FROM"]
    msg["To"] = ", ".join(app.utils.config.CONFIG["EMAIL_TO"])

    # Open the image in binary mode
    with open(save_image_path, "rb") as fp:
        img_data = fp.read()
        msg.add_attachment(
            img_data,
            maintype="image",
            subtype="jpeg",
            filename=save_image_path,
        )

    with smtplib.SMTP_SSL(app.utils.config.CONFIG["EMAIL_SERVER"], app.utils.config.CONFIG["EMAIL_PORT"]) as s:
        s.login(app.utils.config.CONFIG["EMAIL_FROM"], app.utils.config.CONFIG["EMAIL_PASSWORD"])
        s.send_message(msg)
