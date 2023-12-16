from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from .proto import ProtoMemory


class RolloutMemory(ProtoMemory):
    """Generic episode replay class

    Stores batches of data - trajectories / episodes

    Args:
        :param [memory_blueprint]: dictionary containing scheme, groups, max_seq_length and transforms

    Derived State
        :param [data]: SimpleNamespace containing steps given selected arguments
        :param [transforms]: instances of preprocessing transformations
        :param [device]: accelerator device
    """

    def __init__(self, memory_blueprint: dict) -> None:
        super().__init__(memory_blueprint)

    def ensemble_rollout_memory(
        self,
        *,
        data: SimpleNamespace = None,
        device: Optional[str] = "cpu",
    ) -> None:
        """fill data simple namespace with attributes needed for batch creation"""
        super().ensemble_memory(data=data, device=device)

    def update(self, data: dict, time_slice: slice) -> None:
        pass
