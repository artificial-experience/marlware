from types import SimpleNamespace
from typing import Optional
from typing import Tuple

import numpy as np
import torch

from src.util.constants import AttrKey


class ProtoMemory:
    """Prototype Generic rollout memory

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
        """prepare data blueprint given attributes from memory blueprint"""
        if self._transforms is not None:
            # {'actions': ('actions_onehot', [<src.transforms.one_hot.OneHotTransform>])}
            for attribute in self._transforms:
                assert attribute in self._scheme
                new_attribute = self._transforms[attribute][0]
                transforms = self._transforms[attribute][1]

                vshape = self._scheme[attribute][self._data_attr._VALUE_SHAPE.value]
                dtype = self._scheme[attribute][self._data_attr._DTYPE.value]

                # update vshape and dtype information for transform
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                # update internal scheme
                self._scheme[new_attribute] = {
                    self._data_attr._VALUE_SHAPE.value: vshape,
                    self._data_attr._DTYPE.value: dtype,
                }

                if self._data_attr._GROUP.value in self._scheme[attribute]:
                    self._scheme[new_attribute][
                        self._data_attr._GROUP.value
                    ] = self._scheme[attribute][self._data_attr._GROUP.value]

        # add "filled" key to the scheme
        assert (
            self._data_attr._FILLED.value not in self._scheme
        ), '"filled" is a reserved key for masking.'
        self._scheme.update(
            {
                self._data_attr._FILLED.value: {
                    self._data_attr._VALUE_SHAPE.value: (1,),
                    self._data_attr._DTYPE.value: torch.long,
                }
            }
        )

        for field_key, field_info in self._scheme.items():
            assert (
                self._data_attr._VALUE_SHAPE.value in field_info
            ), f"Scheme must define vshape for {field_key}"
            vshape = field_info[self._data_attr._VALUE_SHAPE.value]
            group = field_info.get(self._data_attr._GROUP.value, None)
            dtype = field_info.get(self._data_attr._DTYPE.value, torch.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            shape = vshape
            if group:
                assert (
                    group in self._groups
                ), "Group {} must have its number of members defined in _groups".format(
                    group
                )
                shape = (self._groups[group], *vshape)

            # create transition data scheme
            self._data.transition_data[field_key] = torch.zeros(
                (self._max_seq_length, *shape), dtype=dtype, device=self._device
            )

    def _decode_time_slice(self, item: Tuple[slice, int]):
        """decode information about time slice"""
        decoded = []
        # Ensure item is a list
        assert not isinstance(item, list), "Time slice must be contiguous"
        if isinstance(item, int):
            slice_it = slice(item, item + 1)
            decoded.append(slice_it)
        else:
            decoded.append(item)
        return decoded

    def _new_data_namespace(self):
        """create new data simple namespace"""
        new_data = SimpleNamespace()
        new_data.transition_data = {}
        return new_data

    def _measure_slice_extent(self, indexing_item, max_size):
        """measure how many items given indexing directive"""
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1) // _range[2]

    def _check_safe_view(self, v, dest):
        """check whether its safe to reshape tensor v w.r.t dest shape"""
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError(
                        f"Unsafe reshape of {v.shape} to {dest.shape}"
                    )
            else:
                idx -= 1

    def __repr__(self):
        return "Memory. Max_seq_len:{} Scheme:{} Groups:{}".format(
            self._max_seq_length, self._scheme.keys(), self._groups.keys()
        )
