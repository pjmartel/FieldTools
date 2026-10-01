#!/usr/bin/env python3
"""FieldTools using pytraj to read the trajectory.

Kept for compatibility: equivalent to `python utils/FieldTools.py ... -backend pytraj`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import FieldTools


def python_main(args):
    return FieldTools.main(list(args[1:]) + ["-backend", "pytraj"])


if __name__ == "__main__":
    FieldTools.main(sys.argv[1:] + ["-backend", "pytraj"])
