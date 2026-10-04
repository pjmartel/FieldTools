#!/usr/bin/env python3
"""Compatibility wrapper: same as `fieldtools ... -backend pytraj` (see README.md)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fieldtools.fields import main  # noqa: E402


def python_main(args):
    return main(list(args[1:]) + ["-backend", "pytraj"])


if __name__ == "__main__":
    main(sys.argv[1:] + ["-backend", "pytraj"])
