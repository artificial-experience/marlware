import random
import torch
import numpy as np

from typing import Optional
from logging import Logger

from omegaconf import OmegaConf
from smac.env import StarCraft2Env

from src.cortex import RecQCortex
from src.memory.rollout import RolloutMemory

from src.util.constants import AttrKey


class InteractionWorker:
    """Interface class between environment and memory replay and cortex

    Internal state:
        :param [env]: existing environment instance to be used for interaction
        :param [cortex]: existing multi-agent cortex which is used for action calculation
        :param [mem_replay]: instance of memory replay used for storing trajectories
    """

    def __init__(self) -> None:

        # internal params
        self._env = None
        self._cortex = None
        self._logger = None
        self._device = None
        self._memory_blueprint = None
        self._memory_size = None

        self._data_attr = AttrKey.data
        self._env_attr = AttrKey.env

    def ensemble_interaction_worker(
        self,
        env: StarCraft2Env,
        cortex: RecQCortex,
        memory_blueprint: dict,
        logger: Logger,
        *,
        memory_size: Optional[int] = 1,
        device: Optional[str] = "cpu",
    ) -> None:
        """ensemble interaction worker"""
        self._env = env
        self._cortex = cortex
        self._logger = logger
        self._device = device
        self._memory_blueprint = memory_blueprint
        self._memory_size = memory_size

        # track episode and environment timesteps
        self._episode_ts = 0
        self._env_ts = 0


    def reset(self):
        """set counter to 0, reset env and create new rollout memory"""
        self._episode_ts = 0

        new_rollout_mem = RolloutMemory(self._memory_blueprint)
        new_rollout_mem.ensemble_rollout_memory(memory_size=self._memory_size, device=self._device)
        
        self._env.reset()

        return new_rollout_mem

    def collect_rollout(self, test_mode: bool = False) -> RolloutMemory:
        """collect single episode of data and store in cache"""
        rollout_mem = self.reset()

        terminated = False
        episode_return = 0

        # cortex will operate on a single batch of episodes in order to compute actions
        self._cortex.init_hidden(batch_size=1)

        while not terminated:

            pre_transition_data = {
                self._data_attr._STATE.value: [self._env.get_state()],
                self._data_attr._AVAIL_ACTIONS.value: [self._env.get_avail_actions()],
                self._data_attr._OBS.value: [self._env.get_obs()],
            }

            # update at timestep t
            rollout_mem.update(pre_transition_data, time_slice=self._episode_ts)

            actions = self._cortex.infer_eps_greedy_actions(
                data=rollout_mem, rollout_timestep=self._episode_ts, env_timestep=self._env_ts
            )

            # step the environ
            reward, terminated, env_info = self._env.step(actions[0])

            post_transition_data = {
                self._data_attr._ACTIONS.value: actions,
                self._data_attr._REWARD.value: [(reward,)],
                self._data_attr._TERMINATED.value: [(terminated != env_info.get(self._env_attr._EP_LIMIT.value, False),)],
            }

            # update at timestep t
            rollout_mem.update(post_transition_data, time_slice=self._episode_ts)

            episode_return += reward
            self._episode_ts += 1
            self._env_ts += 1

        # update at timestep t_max + 1
        termination_data = {
            self._data_attr._STATE.value: [self._env.get_state()],
            self._data_attr._AVAIL_ACTIONS.value: [self._env.get_avail_actions()],
            self._data_attr._OBS.value: [self._env.get_obs()],
        }

        rollout_mem.update(termination_data, time_slice=self._episode_ts)

        actions = self._cortex.infer_eps_greedy_actions(
            data=rollout_mem, rollout_timestep=self._episode_ts, env_timestep=self._env_ts
        )

        post_termination_data = {
            self._data_attr._ACTIONS.value: actions,
        }

        rollout_mem.update(post_termination_data, time_slice=self._episode_ts)

        return rollout_mem

    @property
    def environ_timesteps(self):
        """get environment timesteps"""
        return self._env_ts
