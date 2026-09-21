"""Compatibility import for the standalone V7 transport bench."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from animation_modem.v7 import *  # noqa: F401,F403,E402
