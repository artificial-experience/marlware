import random
from types import SimpleNamespace
from typing import Optional, Union
from typing import Tuple

import numpy as np
import torch

from src.util.constants import  AttrKey


class ProtoMemory:
    """Prototype Generic episode replay class

    Stores batches of data - trajectories / episodes

    Args:
        :param [memory_blueprint]: dictionary contatinning scheme, groups, max_seq_length and transforms

    Internal State
        :param [data]: SimpleNamespace containing trajectories given selected arguments
        :param [transforms]: instances of preprocessing transformations
        :param [device]: accelerator device
    """

    def __init__(
        self, memory_blueprint: dict
    ) -> None:

        self._data_attr = AttrKey.data

        self._scheme = memory_blueprint[self._data_attr._SCHEME.value].copy()
        self._groups = memory_blueprint[self._data_attr._GROUP.value].copy()

        # max episode length is always + 1 step to take account of next transitions
        self._max_seq_length = memory_blueprint[self._data_attr._MAX_EP_LEN.value] + 1
        self._transforms = memory_blueprint[self._data_attr._TRANSFORMS.value]

        # internal attrs
        self._data: SimpleNamespace = None
        self._device = None
        self._memory_size = None

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def ensemble_memory(
            self,
            *,
            data: Optional[dict] = None,
            memory_size: Optional[int] = 1,
            device: Optional[str] = "cpu",
            seed: Optional[int] = None,
    ) -> None:
        """fill data simple namespace with attributes needed for batch creation"""
        self._rnd_seed(seed=seed)
        self._device = device
        self._memory_size = memory_size

        if data is not None:
            # get offline data
            self._data = data
        else:
            # prepare new data blueprint
            self._data = SimpleNamespace()
            self._data.transition_data = {}
            self._prepare_data_blueprint()

    def _prepare_data_blueprint(self) -> None:
        """prepare data internal simple namespace attr"""
        if self._transforms is not None:
            # get attributes to transform [ e.g. "actions" ]
            for attr_to_transform in self._transforms:
                assert (
                        attr_to_transform in self._scheme
                ), "attribute to be transformed was not found in scheme"

                new_attr = self._transforms[attr_to_transform][0]
                selected_transforms = self._transforms[attr_to_transform][1]

                vshape = self._scheme[attr_to_transform]["vshape"]
                dtype = self._scheme[attr_to_transform]["dtype"]

                # calculate new vshape and dtype
                for transform in selected_transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                # update scheme of new information
                self._scheme[new_attr] = {
                    "vshape": vshape,
                    "dtype": dtype,
                }

                if "group" in self._scheme[attr_to_transform]:
                    self._scheme[new_attr]["group"] = self._scheme[attr_to_transform][
                        "group"
                    ]

        # ensure filled key is not present in scheme attr
        assert "filled" not in self._scheme, "filled is a reserved key for masking"

        self._scheme.update(
            {
                "filled": {
                    "vshape": (1,),
                    "dtype": torch.long,
                },
            }
        )

        for attr_key, attr_info in self._scheme.items():
            assert (
                    "vshape" in attr_info
            ), f"scheme must define vshape for {attr_key} attr"
            attr_vshape = attr_info["vshape"]
            attr_group = attr_info.get("group", False)
            attr_dtype = attr_info.get("dtype", torch.float32)

            # ensure vshape is tuple
            if isinstance(attr_vshape, int):
                attr_vshape = (attr_vshape,)

            # define shape of attr [ vshape for single, n_agents for groups ]
            shape = attr_vshape
            if attr_group:
                # check if group exists in predefined _groups_ attr
                assert (
                        attr_group in self._groups
                ), "group {} must have its number of agents defined in _groups_".format(
                    attr_group
                )
                shape = (self._groups[attr_group], *attr_vshape)  # 3m - [ 3, vshape ]

            self._data.transition_data[attr_key] = torch.zeros(
                (self._memory_size, self._max_seq_length, *shape),
                dtype=attr_dtype,
                device=self._device,
            )

    def _compute_slices(self, slices: Tuple[slice, slice]) -> list[Union[slice, tuple[slice, slice]]]:
        """compute batch and time slices for update method"""
        computed_slices = []

        # only batch slice given, add full time slice
        if (
                isinstance(slices, slice)  # slice a:b
                or isinstance(slices, int)  # int i
                or (
                isinstance(
                    slices,
                    (
                            list,
                            np.ndarray,
                            torch.LongTensor,
                            torch.cuda.LongTensor,
                    ),  # [ a,b,c ]
                )
        )
        ):
            slices = (slices, slice(None))

        # need the time indexing to be contiguous
        if isinstance(slices[1], list):
            raise IndexError("indexing across Time must be contiguous")

        # compute slices s = ( None, Value, None )
        for s in slices:
            if isinstance(s, int):
                # Convert single indices to slices
                computed_slices.append(slice(s, s + 1))
            else:
                # Leave slices and lists as is
                computed_slices.append(s)
        return computed_slices

    def _compute_num_items(self, computed_slice: slice, horizon: int) -> int:
        """compute number of items to be sliced"""
        if isinstance(computed_slice, list) or isinstance(computed_slice, np.ndarray):
            return len(computed_slice)
        elif isinstance(computed_slice, slice):
            _range = computed_slice.indices(horizon)
            return 1 + (_range[1] - _range[0] - 1) // _range[2]

    def _check_safe_view(self, value: torch.Tensor, destination: torch.Tensor) -> None:
        """ensure reshape / view operation is safe"""
        idx = len(value.shape) - 1
        for s in destination.shape[::-1]:
            if value.shape[idx] != s:
                if s != 1:
                    raise ValueError(
                        "unsafe reshape of {} to {}".format(
                            value.shape, destination.shape
                        )
                    )
            else:
                idx -= 1

    def _new_data_namespace(self) -> SimpleNamespace:
        """create new instance of data namespace"""
        new_data = SimpleNamespace()
        new_data.transition_data = {}
        return new_data

    def __repr__(self):
        return "RolloutMemory. Memory_size:{} Max_seq_len:{} Scheme:{} Groups:{}".format(self._memory_size,
                                                                                     self._max_seq_length,
                                                                                     self._scheme.keys(),
                                                                                     self._groups.keys())
