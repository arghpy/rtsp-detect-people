import json
import sys

import app.utils.logger


def load_json_file(file):
    """Load json file"""
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            content.strip()  # Remove whitespaces
            json_content = json.loads(content)
    except json.JSONDecodeError:
        app.utils.logger.eprint(f"File is not in JSON format: {file}")
        sys.exit(1)
    except FileNotFoundError:
        app.utils.logger.eprint(f"File does not exist: {file}")
        sys.exit(0)
    return json_content
