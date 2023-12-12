from dataclasses import dataclass
from dataclasses import field


@dataclass
class MapConfig:
    prefix: str


@dataclass
class EnvironConfig:
    map: MapConfig
