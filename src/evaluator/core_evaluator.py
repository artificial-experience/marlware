from functools import partial
from logging import Logger
from typing import Optional

import numpy as np
from omegaconf import OmegaConf

from src.cortex import MultiAgentCortex
from src.environ.starcraft import SC2Environ
from src.memory.replay import EpisodeBatch


class CoreEvaluator:
    def __init__(self) -> None:
        self._batch_size = 1
        self._timestep = 0

        # internal state
        self._env = None
        self._env_info = None
        self._mac = None
        self._trace_logger = None

        # horizon
        self._episode_limit = None

    def ensemble_evaluator(
        self, env: SC2Environ, env_info: dict, cortex: MultiAgentCortex, logger: Logger
    ) -> None:
        """instantiate internal states"""
        self._performance = []

        self._env = env
        self._env_info = env_info
        self._mac = cortex
        self._trace_logger = logger

        self._episode_limit = self._env.episode_limit

    def setup(self, scheme, groups, preprocess, device) -> None:
        """setup new batch blueprint"""
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self._batch_size,
            self._episode_limit + 1,
            preprocess=preprocess,
            device=device,
        )

    def reset(self) -> None:
        """reset internal state and create new batch"""
        self._batch = self.new_batch()
        self._env.reset()
        self._timestep = 0

    def evaluate(self, rollout, n_games: Optional[int] = 10):
        """evaluate cortex performance on selected env"""
        evaluation_scores = []
        for game in range(n_games):
            self.reset()

            terminated = False
            episode_return = 0
            self._mac.init_hidden(batch_size=self._batch_size)

            while not terminated:
                pre_transition_data = {
                    "state": [self._env.get_state()],
                    "avail_actions": [self._env.get_avail_actions()],
                    "obs": [self._env.get_obs()],
                }

                self._batch.update(pre_transition_data, ts=self._timestep)

                # Pass the entire batch of experiences up till now to the agents
                # Receive the actions for each agent at this timestep in a batch of size 1
                actions = self._mac.infer_greedy_actions(
                    data=self._batch, rollout_timestep=self._timestep, env_timestep=0
                )

                reward, terminated, env_info = self._env.step(actions[0])
                episode_return += reward

                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [
                        (terminated != env_info.get("episode_limit", False),)
                    ],
                }

                self._batch.update(post_transition_data, ts=self._timestep)

                self._timestep += 1

            last_data = {
                "state": [self._env.get_state()],
                "avail_actions": [self._env.get_avail_actions()],
                "obs": [self._env.get_obs()],
            }
            self._batch.update(last_data, ts=self._timestep)

            # Select actions in the last stored state
            actions = self._mac.infer_greedy_actions(
                data=self._batch, rollout_timestep=self._timestep, env_timestep=0
            )

            self._batch.update({"actions": actions}, ts=self._timestep)

            evaluation_scores.append(episode_return)

        mean_score = np.mean(evaluation_scores)
        self._performance.append(mean_score)
        print(f"rollout: {rollout}, mean_score: {mean_score}")
