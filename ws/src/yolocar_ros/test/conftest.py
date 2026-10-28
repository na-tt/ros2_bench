"""
Make the sibling ``monolith/yolocar`` project importable under colcon test.

The pipeline package is not installed into the ROS workspace (there is no
``pip`` on the Jetson's ``/usr/bin/python3``); the benchmark runs set
``PYTHONPATH=monolith/yolocar/src`` instead. Do the same here so the wrapper's
tests exercise the real pipeline modules.
"""

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
YOLOCAR_SRC = REPO / 'monolith' / 'yolocar' / 'src'

if YOLOCAR_SRC.is_dir() and str(YOLOCAR_SRC) not in sys.path:
    sys.path.insert(0, str(YOLOCAR_SRC))
