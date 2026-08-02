import json
import random
import string
import urllib.request
import urllib.parse
import urllib.error
import getpass
import time
import app.utils.logger


def login(ip, username, password, timeout=5):
    """Logs in and returns (token, expires_at_epoch), or (None, 0) on failure."""
    url = f"http://{ip}/cgi-bin/api.cgi?cmd=Login"
    body = json.dumps([{
        "cmd": "Login",
        "action": 0,
        "param": {"User": {"userName": username, "password": password}}
    }]).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            data = json.loads(raw)
            if "value" not in data[0]:
                app.utils.logger.eprint(f"Login failed, camera response: {raw.decode(errors='replace')}")
                return None, 0
            token_info = data[0]["value"]["Token"]
            expires_at = time.time() + token_info.get("leaseTime", 3600)
            return token_info["name"], expires_at
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        app.utils.logger.eprint(f"Login failed: {e}")
        return None, 0


def get_snapshot(ip, token, channel=0, timeout=5):
    """Fetches a JPEG snapshot using an existing session token."""
    rs = "".join(random.choices(string.ascii_letters + string.digits, k=16))
    params = {"cmd": "Snap", "channel": channel, "rs": rs, "token": token}
    url = f"http://{ip}/cgi-bin/api.cgi?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
            if "image" not in content_type:
                app.utils.logger.eprint(f"Unexpected content type: {content_type} | body: {data[:200]}")
                return None
            return data
    except urllib.error.HTTPError as e:
        app.utils.logger.eprint(f"HTTP error fetching snapshot: {e.code} {e.reason}")
        return None
    except urllib.error.URLError as e:
        app.utils.logger.eprint(f"Connection error fetching snapshot: {e.reason}")
        return None
