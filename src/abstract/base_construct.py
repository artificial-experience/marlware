from abc import ABC
from abc import abstractmethod

import torch


class BaseConstruct(ABC):
    @abstractmethod
    def ensemble_construct(
        self, n_agents: int, n_actions: int, observation_dim: tuple, state_dim: tuple
    ) -> None:
        pass

    @abstractmethod
    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        pass
