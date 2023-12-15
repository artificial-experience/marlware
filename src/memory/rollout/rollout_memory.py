from types import SimpleNamespace

from typing import Dict
from typing import Optional
from typing import Union

import torch

from .proto import ProtoMemory


class RolloutMemory(ProtoMemory):
    """Generic episode replay class

    Stores batches of data - trajectories / episodes

    Args:
        :param [memory_blueprint]: dictionary contatinning scheme, groups, max_seq_length and transforms

    Derived State
        :param [data]: SimpleNamespace containing steps given selected arguments
        :param [transforms]: instances of preprocessing transformations
        :param [device]: accelerator device
    """

    def __init__(
            self, memory_blueprint: dict
    ) -> None:
        super().__init__(memory_blueprint)

    def ensemble_rollout_memory(
            self,
            *,
            data: SimpleNamespace = None,
            memory_size: Optional[int] = 1,
            device: Optional[str] = "cpu",
            seed: Optional[int] = None,
    ) -> None:
        """fill data simple namespace with attributes needed for batch creation"""
        super().ensemble_memory(data=data, memory_size=memory_size, device=device, seed=seed)

    def override_data_device(self, new_device: str) -> None:
        """override data device"""
        for k, v in self._data.transition_data.items():
            self._data.transition_data[k] = v.to(new_device)
        self._device = new_device

    def update(
            self,
            data: Dict[Dict, list],
            batch_slice: Optional[slice] = slice(None),
            time_slice: Optional[slice] = slice(None),
            mark_filled: bool = True,
    ) -> None:
        """update and preprocess data, use slices to handle data in batches"""

        slices = (batch_slice, time_slice)
        slices = self._compute_slices(slices)

        # select target for data update
        for (
                attr,
                value,
        ) in data.items():
            if attr in self._data.transition_data:
                target = self._data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            else:
                raise KeyError(f"{attr} not found in transition nor episode data")

            attr_dtype = self._scheme[attr].get("dtype", torch.float32)
            t_value = value.to(dtype=attr_dtype, device=self._device)

            # try to slice data
            target_slice = target[attr][_slices]
            self._check_safe_view(t_value, target_slice)
            target[attr][_slices] = t_value.view_as(target_slice)

            # preprocess data slice
            if attr in self._transforms:
                new_attr = self._transforms[attr][0]

                value_to_transform = target[attr][_slices]
                for transform in self._transforms[attr][1]:
                    transformed_value = transform.transform(value_to_transform)

                # update target of new transformed value
                target[new_attr][_slices] = transformed_value.view_as(
                    target[new_attr][_slices]
                )

    def max_t_filled(self):
        """check whether the episode is filled"""
        return torch.sum(self._data.transition_data["filled"], 1).max(0)[0]

    def __getitem__(
            self, item: Union[str, tuple]
    ) -> Union["RolloutMemory", tuple]:
        """based on type return item from data simple namespace"""
        if isinstance(item, str):
            # get interaction batches
            if item in self._data.transition_data:
                return self._data.transition_data[item]
            else:
                raise ValueError(
                    "{} item could not be found in episode or transition data".format(
                        item
                    )
                )

        else:
            # get training batches
            slices = self._compute_slices(item)
            new_data = self._new_data_namespace()
            for attr, value in self._data.transition_data.items():
                new_data.transition_data[attr] = value[slices]

            new_max_seq_length = self._compute_num_items(
                slices[1], self._max_seq_length
            )

            # update max seq_length
            new_memory_blueprint = {
                self._data_attr._SCHEME.value: self._scheme,
                self._data_attr._GROUP.value: self._groups,
                self._data_attr._MAX_EP_LEN.value: new_max_seq_length,
                self._data_attr._TRANSFORMS.value: self._transforms,
            }

            new_batch = RolloutMemory(
                new_memory_blueprint
            )
            new_batch.ensemble_rollout_memory(data=new_data, device=self._device)
            return new_batch
