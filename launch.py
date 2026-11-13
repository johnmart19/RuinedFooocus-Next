"""Launch RuinedFooocus from a clone or extracted distribution."""

from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parent)
from ruinedfooocus.app import bootstrap  # noqa: E402,F401
