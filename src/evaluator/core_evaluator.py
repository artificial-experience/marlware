from collections import defaultdict
from typing import Any
from typing import Optional

import numpy as np
import ray

from src.worker import InteractionWorker


@ray.remote
class CoreEvaluator:
    def __init__(self, worker: InteractionWorker) -> None:
        self._worker = worker

    def ensemble_evaluator(self) -> None:
        """ensemble internal variables"""
        self._mean_performance = []
        self._best_score = 0.0
        self._highest_battle_win_score = 0.0

        # set metrics mapping
        self._metrics = defaultdict(int)

    def evaluate(
        self,
        rollout: int,
        n_games: Optional[int] = 10,
        replay_save_freq: Optional[int] = 10,
    ) -> tuple[bool, defaultdict[Any, int]]:
        """evaluate current model on n_games"""
        evaluation_scores = []
        won_battles = []

        for game_idx in range(n_games):
            # save replay
            if game_idx % replay_save_freq == 0:
                worker_output_ref = self._worker.collect_rollout.remote(
                    test_mode=True, save_replay=True
                )
            else:
                worker_output_ref = self._worker.collect_rollout.remote(
                    test_mode=True, save_replay=False
                )

            worker_output = ray.get(worker_output_ref)
            metrics = worker_output[1]

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
        self._update_metrics(rollout, mean_performance, mean_won_battles)

        return is_new_best, self._metrics

    def _update_metrics(
        self, rollout: int, mean_scores: list, mean_won_battles: list
    ) -> None:
        """log metrics using logger"""
        self._metrics["mean_performance"] = self._mean_performance
        self._metrics["mean_scores"] = mean_scores
        self._metrics["mean_won_battles"] = mean_won_battles
        self._metrics["best_score"] = self._best_score
        self._metrics["highest_battle_win_score"] = self._highest_battle_win_score
        self._metrics["rollout"] = rollout
