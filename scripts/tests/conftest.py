"""
This is the file to resolve the directory structure.
Also helpful to mock and write cases.
"""

import sys
import os
from unittest.mock import MagicMock


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


MODULES_TO_STUB = [
    "datasets",
    "ftfy",
    "tqdm",
    "tqdm.auto",
    "google",
    "google.cloud",
    "google.cloud.storage",
]

for mod in MODULES_TO_STUB:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()