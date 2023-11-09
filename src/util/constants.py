import os
from enum import Enum
from pathlib import Path


class Directories(Enum):
    MARLROOT_DIR = Path(os.getenv("MARLROOT", ".")) / "src"
    CONF_DIR = MARLROOT_DIR / "conf"
    TRAINABLE_CONFIG_DIR = CONF_DIR / "trainable"
