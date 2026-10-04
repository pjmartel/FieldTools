#!/usr/bin/env python3
"""Compatibility wrapper: same as the `fieldtools` command (see README.md)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fieldtools.fields import *  # noqa: E402,F401,F403
from fieldtools.fields import main, python_main  # noqa: E402,F401

if __name__ == "__main__":
    main()
