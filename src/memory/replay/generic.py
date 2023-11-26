import random
from functools import partialmethod
from types import SimpleNamespace
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

from src.util import constants


class GenericEpisodeReplay:
    """Generic episode replay class

    Stores batches of data - trajectories / episodes

    Args:
        :param [scheme]: schema used to create data template
        :param [groups]: number of agents
        :param [mem_size]: size of a batch for sampling method
        :param [max_seq_length]: maximum number of trajectories within episode

    Internal State
        :param [data]: SimpleNamespace containing trajectories given selected arguments
        :param [transforms]: instances of preprocessing transformations
        :param [device]: accelerator device
    """

    def __init__(
        self, scheme: dict, groups: dict, mem_size: int, max_seq_length: int
    ) -> None:
        self._scheme = scheme
        self._groups = groups
        self._mem_size = mem_size
        self._max_seq_length = max_seq_length

        # internal attrs
        self._data: SimpleNamespace = None
        self._transforms = None
        self._device = None

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def ensemble_episode_memory(
        self,
        *,
        data: Optional[dict] = None,
        transforms: Optional[dict] = None,
        device: Optional[str] = "cpu",
        seed: Optional[int] = None
    ) -> None:
        """fill data simple namespace with attributes needed for batch creation"""
        self._rnd_seed(seed=seed)

        self._transforms = transforms
        self._device = device

        if data is not None:
            # get offline data
            self._data = data
        else:
            # prepare new data blueprint
            self._data = SimpleNamespace()
            self._data.transition_data = {}
            self._data.episode_data = {}
            self._prepare_data_blueprint()

        # track the memory and number of episodes in replay memory
        self._mem_index = 0
        self._episodes_in_mem = 0

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

                if "episode_const" in self._scheme[attr_to_transform]:
                    self._scheme[new_attr]["episode_const"] = self._scheme[
                        attr_to_transform
                    ]["episode_const"]

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
            attr_episode_const = attr_info.get("episode_const", False)
            attr_group = attr_info.get("group", False)
            attr_dtype = attr_info.get("dtype", torch.float32)

            # ensure vshape is tuple
            if isinstance(attr_vshape, int):
                vshape = (vshape,)

            # define shape of attr [ vshape for single, n_agents for groups ]
            shape = vshape
            if attr_group:
                # check if group exists in predefined _groups_ attr
                assert (
                    group in self._groups
                ), "group {} must have its number of agents defined in _groups_".format(
                    attr_group
                )
                shape = (self._groups[attr_group], *vshape)  # 3m - [ 3, vshape ]

            # for most starcraft envs the episode len is not const
            if attr_episode_const:
                self._data.episode_data[attr_key] = torch.zeros(
                    (self._mem_size, *shape), dtype=attr_dtype, device=self._device
                )
            else:
                self._data.transition_data[attr_key] = torch.zeros(
                    (self._mem_size, self._max_seq_length, *shape),
                    dtype=attr_dtype,
                    device=self._device,
                )

    def _compute_slices(self, slices: Tuple[slice, slice]) -> Tuple[slice, slice]:
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
            _range = computed_slices.indices(horizon)
            return 1 + (_range[1] - _range[0] - 1) // _range[2]

    def _check_safe_view(self, value: list, destination: torch.Tensor) -> None:
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

    def _new_data_namespace(self) -> None:
        """create new instance of data namespace"""
        new_data = SimpleNamespace()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def override_data_device(self, new_device: str) -> None:
        """override data device"""
        for k, v in self._data.transition_data.items():
            self._data.transition_data[k] = v.to(new_device)
        for k, v in self._data.episode_data.items():
            self._data.episode_data[k] = v.to(new_device)
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
            elif attr in self._data.episode_data:
                target = self._data.episode_data
                _slices = slices[0]
            else:
                raise KeyError(
                    f"{attr} not found in transition not episode data"
                )

            attr_dtype = self._scheme[attr].get("dtype", torch.float32)
            t_value = torch.tensor(value, dtype=attr_dtype, device=self._device)

            # try to slice data
            target_slice = target[attr][_slices]
            self._check_safe_view(t_value, target_slice)
            target[attr][_slices] = t_value.view_as(traget_slice)

            # preprocess data slice
            if attr in self._transforms:
                new_attr = self._transforms[attr][0]

                value_to_transform = target[attr][_slices]
                for transform in self._transforms[attr][0]:
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
    ) -> Union["GenericEpisodeReplay", tuple]:
        """based on type return item from data simple namespace"""
        if isinstance(item, str):
            if item in self._data.episode_data:
                return self._data.episode_data[item]
            elif item in self._data.transition_data:
                return self._data.transition_data[item]
            else:
                raise ValueError(
                    "{} item could not be found in episode or transition data".format(
                        item
                    )
                )

        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_namespace()
            for key in item:
                if key in self._data.transition_data:
                    new_data.transition_data[key] = self._data.transition_data[key]
                elif key in _self._data.episode_data:
                    new_data.episode_data[key] = self._data.episode_data[key]
                else:
                    raise KeyError(f"unrecognised key {key}")

            # update the scheme to only have the requested keys
            new_scheme = {key: self._scheme[key] for key in item}
            new_groups = {
                self._scheme[key]["group"]: self._groups[self._scheme[key]["group"]]
                for key in item
                if "group" in self._scheme[key]
            }

            # create new batch and return
            new_batch = GenericEpisodeReplay(
                scheme=new_scheme,
                groups=new_groups,
                mem_size=self._mem_size,
                max_seq_length=self._max_seq_length,
            )
            new_batch.ensemble_episode_memory(data=new_data, device=self._device)
            return new_batch

        else:
            slices = self._compute_slices(item)
            new_data = self._new_data_namespace()
            for attr, value in self._data.transition_data.items():
                new_data.transition_data[attr] = value[slices]
            for attr, value in self._data.episode_data.items():
                new_data.episode_data[attr] = value[slices[0]]

            new_mem_size = self._compute_num_items(slices[0], self._mem_size)
            new_max_seq_length = self._compute_num_items(
                slices[1], self._max_seq_length
            )

            new_batch = GenericEpisodeReplay(
                scheme=self._scheme,
                groups=self._groups,
                mem_size=new_mem_size,
                max_seq_length=new_max_seq_length,
            )
            new_batch.ensemble_episode_memory(data=new_data, devive=self._device)
            return new_batch

    # TODO: not finished yet make sure you understand that shit
    def memorize_episode(self, episode: "GenericEpisodeReplay") -> None:
        """memorize patch of trajectories"""
        if self._mem_index + episode._mem_size <= self._mem_size:
            self.update(
                episode._data.transition_data,
                slice(self._mem_index, self._mem_index + episode._batch_size),
                slice(0, episode._max_seq_length),
                mark_filled=False,
            )
            self.update(
                episode._data.episode_data,
                slice(self._mem_index, self._mem_index + episode._batch_size),
            )
            self._mem_index = self._mem_index + episode._batch_size
            self._episodes_in_mem = max(self._episodes_in_mem, self._mem_index)
            self._mem_index = self._mem_index % self._mem_size
            assert self._mem_index < self._mem_size
        else:
            buffer_left = self._mem_size - self._mem_index
            self.memorize_episode(episode[0:buffer_left, :])
            self.memorize_episode(episode[buffer_left:, :])

    def can_sample(self, batch_size: int) -> bool:
        """checks whether memory is ready"""
        return self._episodes_in_mem >= batch_size

    def sample(self, batch_size: int) -> dict:
        """sample batch of memories"""
        assert self.can_sample(batch_size), "{} batch size could not be sampled".format(
            batch_size
        )

        # use __getitem__ method to return memory batch
        if self._episodes_in_mem == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self._episodes_in_mem, batch_size, replace=False)
            return self[ep_ids]

    def __repr__(self):
        return "GenericEpisodeReplay. {}/{} Episodes. Keys:{} Groups:{}".format(
            self._episodes_in_mem,
            self._mem_size,
            self._scheme.keys(),
            self._groups.keys(),
        )

    # ---- ---- ---- ---- ---- --- ---- #
    # --- Internal Partial Methods ---- #
    # ---- ---- ---- ---- ---- --- ---- #
