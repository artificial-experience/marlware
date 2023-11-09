from typing import TypeVar

from src.abstract import BaseConstruct

TrainableComponent = TypeVar("TrainableComponent", bound=BaseConstruct)

from .base_qmix import BaseQMIX
