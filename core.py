import sys
from datetime import datetime


def eprint(s):
    """Print to stderr with current time"""
    print(f"{datetime.now()}: {s}", file=sys.stderr, flush=True)


def pprint(s):
    """Print to stdout with current time"""
    print(f"{datetime.now()}: {s}", file=sys.stdout, flush=True)
