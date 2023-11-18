from typing import TypeVar

from src.abstract import ProtoTrainable

Trainable = TypeVar("Trainable", bound=ProtoTrainable)

from .qmix_core import QmixCore
