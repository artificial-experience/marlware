from typing import Optional

import numpy as np

from src.memory.rollout import RolloutMemory


class RolloutCluster(RolloutMemory):
    """
    Memorize data and allow for sampling

    Args:
        :param [memory_blueprint]: dictionary contatinning scheme, groups, max_seq_length and transforms

    """
    def __init__(
        self, memory_blueprint: dict
    ) -> None:

        super().__init__(memory_blueprint=memory_blueprint)

        self._max_rollouts_in_mem = None

    def ensemble_rollout_cluster(
        self,
        max_rollouts_in_mem: int,
        *,
        device: Optional[str] = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        super().ensemble_rollout_memory(memory_size=max_rollouts_in_mem, device=device, seed=seed)
        self._max_rollouts_in_mem = max_rollouts_in_mem

        # keep track of episodes in cluster and buffer index
        self._buffer_index = 0
        self._episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self._buffer_index + ep_batch._memory_size <= self._max_rollouts_in_mem:
            self.update(
                ep_batch._data.transition_data,
                slice(self._buffer_index, self._buffer_index + ep_batch._memory_size),
                slice(0, ep_batch._max_seq_length),
                mark_filled=False,
            )
            self._buffer_index = self._buffer_index + ep_batch._memory_size
            self._episodes_in_buffer = max(self._episodes_in_buffer, self._buffer_index)
            self._buffer_index = self._buffer_index % self._max_rollouts_in_mem
            assert self._buffer_index < self._max_rollouts_in_mem
        else:
            buffer_left = self._max_rollouts_in_mem - self._buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self._episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self._episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(
                self._episodes_in_buffer, batch_size, replace=False
            )
            return self[ep_ids]
