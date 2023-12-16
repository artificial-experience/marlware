from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from src.util.constants import AttrKey


class ProtoMemory:
    """Prototype Generic episode replay class

    Stores batches of data - trajectories / episodes

    Args:
        :param [memory_blueprint]: dictionary containing scheme, groups, max_seq_length and transforms

    Internal State
        :param [data]: SimpleNamespace containing trajectories given selected arguments
        :param [transforms]: instances of preprocessing transformations
        :param [device]: accelerator device
    """

    def __init__(self, memory_blueprint: dict) -> None:
        self._data_attr = AttrKey.data

        self._scheme = memory_blueprint[self._data_attr._SCHEME.value].copy()
        self._groups = memory_blueprint[self._data_attr._GROUP.value].copy()

        # max episode length is always + 1 step to take account of next transitions
        self._max_seq_length = memory_blueprint[self._data_attr._MAX_SEQ_LEN.value] + 1
        self._transforms = memory_blueprint[self._data_attr._TRANSFORMS.value]

        # internal attrs
        self._data: SimpleNamespace = None
        self._device = None

    def ensemble_memory(
        self,
        *,
        data: Optional[dict] = None,
        device: Optional[str] = "cpu",
    ) -> None:
        """fill data simple namespace with attributes needed for batch creation"""
        self._device = device

        if data is not None:
            # get offline data
            self._data = data
        else:
            # prepare new data blueprint
            self._data = SimpleNamespace()
            self._data.transition_data = {}
            self._prepare_data_blueprint()

    def _prepare_data_blueprint(self):
        pass

    def __repr__(self):
        return "Memory. Max_seq_len:{} Scheme:{} Groups:{}".format(
            self._max_seq_length, self._scheme.keys(), self._groups.keys()
        )
