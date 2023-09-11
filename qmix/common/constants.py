import os
from enum import Enum
from pathlib import Path


class Directories(Enum):
    MARLROOT_DIR = Path(os.getenv("MARLROOT", "."))
    CONFIG_DIR = MARLROOT_DIR / "config"
    TRAINABLE_CONFIG_DIR = CONFIG_DIR / "trainable"
    ENVIRONMENT_CONFIG_DIR = CONFIG_DIR / "environment"
