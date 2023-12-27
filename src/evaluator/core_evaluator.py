from functools import partial
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np

from src.environ.starcraft import SC2Environ
from src.worker import InteractionWorker


class CoreEvaluator:
    def __init__(
        self, env: SC2Environ, worker: InteractionWorker, logger: Logger
    ) -> None:
        self._env = env
        self._worker = worker
        self._logger = logger

    def ensemble_evaluator(self, replay_save_path: Path) -> None:
        self._env.replay_dir = replay_save_path

        self._mean_performance = []
        self._best_score = 0.0
        self._highest_battle_win_score = 0.0

    def evaluate(self, rollout: int, n_games: Optional[int] = 10) -> bool:
        """evaluate current model on n_games"""
        evaluation_scores = []
        won_battles = []

        for game_idx in range(n_games):
            _, metrics = self._worker.collect_rollout(test_mode=True)

            # parse metrics
            score: int = metrics["evaluation_score"]
            battle_won: int = metrics["evaluation_battle_won"]

            # append containers
            evaluation_scores.append(score)
            won_battles.append(battle_won)

        # get mean performance
        mean_performance = np.mean(evaluation_scores)
        mean_won_battles = np.mean(won_battles)

        # update internal state
        self._mean_performance.append(mean_performance)
        self._highest_battle_win_score = max(
            self._highest_battle_win_score, mean_won_battles
        )

        is_new_best = True if (mean_performance > self._best_score) else False
        if is_new_best:
            self._best_score = mean_performance

        # log metrics
        self._log_metrics(rollout, mean_performance, mean_won_battles)

        return is_new_best

    def _log_metrics(
        self, rollout: int, mean_scores: list, mean_won_battles: list
    ) -> None:
        """log metrics using logger"""
        if not self._mean_performance:
            return

        self._logger.log_stat("eval_score_mean", mean_scores, rollout)

        # Calculate statistics
        eval_running_mean = np.mean(self._mean_performance)
        eval_score_std = np.std(self._mean_performance)

        # Log running mean and standard deviation
        self._logger.log_stat("eval_score_running_mean", eval_running_mean, rollout)
        self._logger.log_stat("eval_score_std", eval_score_std, rollout)
        self._logger.log_stat("eval_won_battles_mean", mean_won_battles, rollout)
        self._logger.log_stat(
            "eval_most_won_battles", self._highest_battle_win_score, rollout
        )
        self._logger.log_stat("eval_mean_higest_score", self._best_score, rollout)

        # Calculate and log the variation between the most recent two evaluations, if available
        if len(self._mean_performance) >= 2:
            # Calculate the variation between the mean scores of the last two evaluations
            recent_mean_scores = [
                np.mean(sublist) for sublist in self._mean_performance[-2:]
            ]
            eval_score_var = np.abs(recent_mean_scores[-1] - recent_mean_scores[-2])
            self._logger.log_stat("eval_score_var", eval_score_var, rollout)
