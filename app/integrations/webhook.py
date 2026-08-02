from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import app.utils.logger
import json


alert = False
payload = None

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global alert, payload
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        payload = body.decode(errors="replace")
        try:
            data = json.loads(body)
            if data.get("alarm", {}).get("type") == "PEOPLE":
                alert = True
                payload = data
        except json.JSONDecodeError:
            print("Non-JSON payload received:", body.decode(errors="replace"))
        self.send_response(200)
        self.end_headers()

def camera_alert():
    global alert
    if alert:
        app.utils.logger.pprint("Webhook received, person detected")
        alert = False
        return True
    return False
