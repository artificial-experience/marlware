import random
from typing import Optional
from typing import Tuple

import numpy as np
import torch

from src.memory.shard import MemoryShard
from src.util.constants import AttrKey


class Memory:
    """
    Memory class meant to be used as an interface between shard and cluster

    Args:
        param: [memories]: memories chosen from cluster
    """

    def __init__(self, memories: np.ndarray) -> None:
        self._memories = memories
        self._data_attr = AttrKey.data

    @classmethod
    def from_data(cls, data: np.ndarray):
        """
        Class method to create a Memory instance from raw data.
            :param [data]: Raw data to be processed and used for creating Memory instance.
            :return: Memory instance.
        """
        # Process raw data into a format suitable for Memory
        processed_data = cls._process_data(data)

        # Create a Memory instance with processed data
        return cls(processed_data)

    @staticmethod
    def _process_data(data: np.ndarray):
        """
        Static method to process raw data.
            :param [data]: Raw data to be processed.
            :return: Processed data.
        """
        # Placeholder for data processing logic
        # Modify this as needed to process your data
        processed_data = data
        return processed_data

    def max_t_filled(self) -> torch.Tensor:
        """calculate maximum filled timesteps across all memories"""
        max_filled = max(
            torch.sum(
                memory._data.transition_data[self._data_attr._FILLED.value], 0
            ).max()
            for memory in self._memories
        )
        return max_filled.item()

    def override_data_device(self, device: str):
        """move memory shards into certain device"""
        for memory in self._memories:
            memory.move_to_device(device)

    def __getitem__(self, item):
        """logic for batch and time slicing"""
        # return torch tensor consisting of trainable memories
        if isinstance(item, str):
            # Collect trainable data tensors
            trainable_data = [
                memory[item]
                for memory in self._memories
                if isinstance(memory[item], torch.Tensor)
            ]

            # Check if trainable data is not empty
            if not trainable_data:
                raise ValueError("No trainable data found for the given item.")

            # Stack or concatenate tensors
            # Use torch.stack if all tensors have the same shape, else use torch.cat
            return (
                torch.stack(trainable_data)
                if all(
                    tensor.shape == trainable_data[0].shape for tensor in trainable_data
                )
                else torch.cat(trainable_data, dim=0)
            )

        # return new memory object that has modified memories ( by slices )
        else:
            bs, ts = self._decode_slice_information(item)
            new_memories = self._memories.copy()
            for idx, memory in enumerate(self._memories):
                new_memory = memory[ts]
                new_memories[idx] = new_memory

            # safe approach
            new_mem_obj = Memory.from_data(new_memories)
            return new_mem_obj

    def _decode_slice_information(self, item):
        """decode information about batch and time slice"""
        decoded = []
        assert not isinstance(item[0], list), "Time slice must be contiguous"
        for it in item:
            if isinstance(it, int):
                decoded.append(slice(it, it + 1))
            else:
                decoded.append(it)
        return decoded

    def __repr__(self):
        return f"Memory. Number of memories sampled: {len(self._memories)}"


class MemoryCluster:
    """
    Memorize data and allow for sampling

    Args:
        :param [mem_size]: max number of shards in memory cluster

    """

    def __init__(self, mem_size: int) -> None:
        self._mem_size = mem_size
        self._data_attr = AttrKey.data

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    # TODO: add support for other sampling methods
    def ensemble_memory_cluster(
        self,
        *,
        device: Optional[str] = "cpu",
        sampling_method: Optional[str] = "uniform",
        seed: Optional[int] = None,
    ) -> None:
        """set memory pointers and sampling strategy for the cluster"""
        self._rnd_seed(seed=seed)

        # number of episodes to sample
        # internal attributes for sampling
        self._sampling_method = sampling_method
        self._mem_pointer = 0

        self._shards = np.empty(self._mem_size, dtype=object)

    def insert_memory_shard(self, mem_shard: MemoryShard) -> None:
        """insert memory shard into cluster"""
        mem_index = self._mem_pointer % self._mem_size
        self._shards[mem_index] = mem_shard
        self._mem_pointer += 1

    def can_sample(self, batch_size: int) -> bool:
        """check if there is enough memories to sample from"""
        return self._mem_pointer >= batch_size

    def sample(self, batch_size: int):
        """Sample batch size of memories"""
        assert self.can_sample(batch_size)

        # Filter out empty memory shards
        non_zero_mask = np.array([shard is not None for shard in self._shards])
        non_zero_shards = self._shards[non_zero_mask]

        if len(non_zero_shards) < batch_size:
            raise ValueError(
                "Not enough non-zero shards to sample the requested batch size."
            )

        chosen_shards = None
        if self._sampling_method == "uniform":
            chosen_shards = np.random.choice(non_zero_shards, batch_size, replace=False)
        else:
            raise KeyError(
                f"{self._sampling_method} sampling method is not implemented"
            )

        memories = Memory.from_data(chosen_shards)
        return memories

    def __repr__(self):
        return "MemoryCluster. {}/{} memory shards".format(
            self._mem_pointer, self._mem_size
        )
