from abc import ABC
from abc import abstractmethod

import torch


class ProtoTrainable(ABC):
    @abstractmethod
    def ensemble_trainable(
        self, n_agents: int, n_actions: int, observation_dim: tuple, state_dim: tuple
    ) -> None:
        pass

    @abstractmethod
    def calculate_loss(self, batch: torch.Tensor) -> torch.Tensor:
        pass
