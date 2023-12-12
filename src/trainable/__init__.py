from typing import TypeVar

from src.abstract import ProtoTrainable

Trainable = TypeVar("Trainable", bound=ProtoTrainable)

from .one_step_qmix import OneStepQmix
