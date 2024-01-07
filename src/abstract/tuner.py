from abc import ABC
from abc import abstractmethod
from logging import Logger
from typing import Optional

import numpy as np


class ProtoTuner(ABC):
    @abstractmethod
    def commit(
        self,
        env_conf: str,
        accelerator: str,
        logger: Logger,
        *,
        seed: Optional[int] = None,
    ) -> None:
        pass

    @abstractmethod
    def optimize(
        self,
        n_timesteps: int,
        batch_size: int,
        eval_schedule: int,
        checkpoint_freq: int,
        eval_n_games: int,
        display_freq: int,
    ) -> np.ndarray:
        pass
