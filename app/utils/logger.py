from datetime import datetime
import sys


def eprint(s):
    """Print to stderr with current time"""
    print(f"{datetime.now()}: [WARN] {s}", file=sys.stderr, flush=True)


def pprint(s):
    """Print to stdout with current time"""
    print(f"{datetime.now()}: [INFO] {s}", file=sys.stdout, flush=True)
