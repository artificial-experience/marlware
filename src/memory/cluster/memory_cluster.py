import random
from typing import Optional

import numpy as np
import torch

from src.memory.shard import MemoryShard


class MemoryCluster:
    """
    Memorize data and allow for sampling

    Args:
        :param [mem_size]: max number of shards in memory cluster

    """

    def __init__(self, mem_size: int) -> None:
        self._mem_size = mem_size

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
        seed: Optional[int] = None
    ) -> None:
        """set memory pointers and sampling strategy for the cluster"""
        self._rnd_seed(seed=seed)

        # number of episodes to sample
        # internal attributes for sampling
        self._sampling_method = sampling_method
        self._mem_pointer = 0
        self._mem_shards_in_cluster = 0

    def insert_memory_shard(self, mem_shard: MemoryShard) -> None:
        """insert memory shard into cluster"""
        print(mem_shard)

    def can_sample(self, batch_size: int) -> bool:
        """check if there is enough memories to sample from"""
        return self._mem_shards_in_cluster >= batch_size

    def sample(self, batch_size: int):
        """sample batch size of memories"""
        assert self.can_sample(batch_size)
        if self._sampling_method == "uniform":
            if self._mem_shards_in_cluster == batch_size:
                return
            else:
                episode_ids = np.random.choice(
                    self._mem_shards_in_cluster, batch_size, replace=False
                )
                return
        else:
            raise KeyError(
                f"{self._sampling_method} sampling method is not implemented"
            )

    def __repr__(self):
        return "MemoryCluster. {}/{} memory shards".format(
            self._mem_shards_in_cluster, self._mem_size
        )
