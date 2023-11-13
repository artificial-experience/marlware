from typing import TypeVar

from src.abstract import ProtoTrainable

Trainable = TypeVar("Trainable", bound=ProtoTrainable)

from .base_qmix import BaseQMIX
