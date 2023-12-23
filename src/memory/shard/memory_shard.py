from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from .proto import ProtoMemory


class MemoryShard(ProtoMemory):
    """Single rollout memory shard

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
        # keep memory blueprint for later use
        self._memory_blueprint = memory_blueprint

    def ensemble_memory_shard(
        self,
        *,
        data: SimpleNamespace = None,
        device: Optional[str] = "cpu",
    ) -> None:
        """fill data simple namespace with attributes needed for batch creation"""
        super().ensemble_memory(data=data, device=device)

    def __getitem__(self, item):
        """either return concrete data or slice w.r.t time and return"""
        # 1. item is string (e.g. "state")
        if isinstance(item, str):
            if item in self._data.transition_data:
                return self._data.transition_data[item]
        # 2. item is time slice (e.g. (1,2,None))
        else:
            time_slice = self._decode_time_slice(item)
            new_data = self._new_data_namespace()
            for k, v in self._data.transition_data.items():
                new_data.transition_data[k] = v[time_slice]

            # recalculate max_seq_length param
            max_time = self._measure_slice_extent(time_slice, self._max_seq_length)

            # create new MemoryShard instance
            memory_blueprint = self._memory_blueprint.copy()
            memory_blueprint[self._data_attr._MAX_SEQ_LEN.value] = max_time
            memory_shard = MemoryShard(memory_blueprint)
            memory_shard.ensemble_memory_shard(data=new_data, device=self._device)
            return memory_shard

    def update(self, data: dict, time_slice: slice, mark_filled=True) -> None:
        """update internal data namespace with new data"""
        time_slice = self._decode_time_slice(time_slice)
        for attr_key, attr_value in data.items():
            if attr_key in self._data.transition_data:
                target = self._data.transition_data
                if mark_filled:
                    target[self._data_attr._FILLED.value][time_slice] = 1
                    mark_fileld = False
                _time_slice = time_slice
            else:
                raise KeyError(f"{attr_key} not found in transition data")

            dtype = self._scheme[attr_key].get(
                self._data_attr._DTYPE.value, torch.float32
            )
            value = torch.tensor(attr_value, dtype=dtype, device=self._device)
            self._check_safe_view(value, target[attr_key][_time_slice])
            target[attr_key][_time_slice] = value.view_as(target[attr_key][_time_slice])

            if attr_key in self._transforms:
                new_attr_key = self._transforms[attr_key][0]
                value = target[attr_key][_time_slice]
                for transform in self._transforms[attr_key][1]:
                    value = transform.transform(value)
                target[new_attr_key][_time_slice] = value.view_as(
                    target[new_attr_key][_time_slice]
                )
