from functools import partial
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np

from src.environ.starcraft import SC2Environ
from src.worker import InteractionWorker


class CoreEvaluator:
    def __init__(self, n_games: int) -> None:
        self._n_games = n_games

    def ensemble_evaluator(self, env: SC2Environ, worker: InteractionWorker) -> None:
        """instantiate internal states"""
        pass
