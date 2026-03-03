import sys
import json
from datetime import datetime


def eprint(s):
    """Print to stderr with current time"""
    print(f"{datetime.now()}: {s}", file=sys.stderr, flush=True)


def pprint(s):
    """Print to stdout with current time"""
    print(f"{datetime.now()}: {s}", file=sys.stdout, flush=True)


def load_json_file(file):
    """Load json file"""
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            content.strip()  # Remove whitespaces
            json_content = json.loads(content)
    except json.JSONDecodeError:
        eprint(f"File is not in JSON format: {file}")
        sys.exit(1)
    except FileNotFoundError:
        eprint(f"File does not exist: {file}")
        sys.exit(0)
    return json_content
