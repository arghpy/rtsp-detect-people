import json
import sys

from app.utils.logger import eprint


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
