from abc import ABC
from abc import abstractmethod
from typing import Dict

import numpy as np
import torch


class ProtoCortex(ABC):
    @abstractmethod
    def ensemble_cortex(
        self, n_agents: int, n_actions: int, observation_dim: tuple, state_dim: tuple
    ) -> None:
        pass

    @abstractmethod
    def infer_actions(
        self,
        data: dict,
        rollout_timestep: int,
        env_timestep: int,
        evaluate: bool = False,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def estimate_q_vals(
        self, feed: Dict[str, torch.Tensor], use_target: bool = False
    ) -> torch.Tensor:
        pass
