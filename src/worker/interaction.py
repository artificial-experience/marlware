from collections import defaultdict
from logging import Logger
from typing import Optional
from typing import Tuple

import ray
from smac.env import StarCraft2Env

from src.cortex import RecQCortex
from src.memory.shard import MemoryShard
from src.util.constants import AttrKey


@ray.remote
class InteractionWorker:
    """Interface class between environment and memory replay and cortex

    Internal state:
        :param [env]: existing environment instance to be used for interaction
        :param [cortex]: existing multi-agent cortex which is used for action calculation
        :param [mem_replay]: instance of memory replay used for storing trajectories
    """

    def __init__(self) -> None:
        self._data_attr = AttrKey.data
        self._env_attr = AttrKey.env

        # internal params
        self._env = None
        self._cortex = None
        self._logger = None
        self._device = None
        self._memory_blueprint = None

    def ensemble_interaction_worker(
        self,
        env: StarCraft2Env,
        cortex: RecQCortex,
        memory_blueprint: dict,
        logger: Logger,
        *,
        device: Optional[str] = "cpu",
    ) -> None:
        """ensemble interaction worker"""
        self._env = env
        self._cortex = cortex
        self._logger = logger
        self._device = device
        self._memory_blueprint = memory_blueprint

        # track episode and environment timesteps
        self._episode_ts = 0
        self._env_ts = 0

    def reset(self):
        """set counter to 0, reset env and create new rollout memory"""
        self._episode_ts = 0

        new_mem_shard = MemoryShard(memory_blueprint=self._memory_blueprint)
        new_mem_shard.ensemble_memory_shard(device=self._device)

        self._env.reset()

        return new_mem_shard

    def collect_rollout(self, test_mode: bool = False) -> Tuple[MemoryShard, dict]:
        """collect single episode of data and store in cache"""
        memory_shard = self.reset()

        metrics = defaultdict(int)
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
            memory_shard.update(pre_transition_data, time_slice=self._episode_ts)

            # Conditional action inference based on test_mode
            if test_mode:
                actions = self._cortex.infer_greedy_actions(
                    data=memory_shard,
                    rollout_timestep=self._episode_ts,
                    env_timestep=-1,
                )
            else:
                actions = self._cortex.infer_eps_greedy_actions(
                    data=memory_shard,
                    rollout_timestep=self._episode_ts,
                    env_timestep=self._env_ts,
                )

            # step the environ
            reward, terminated, env_info = self._env.step(actions[0])

            post_transition_data = {
                self._data_attr._ACTIONS.value: actions,
                self._data_attr._REWARD.value: [(reward,)],
                self._data_attr._TERMINATED.value: [
                    (terminated != env_info.get(self._env_attr._EP_LIMIT.value, False),)
                ],
            }

            # update at timestep t
            memory_shard.update(post_transition_data, time_slice=self._episode_ts)

            episode_return += reward
            battle_won = int(env_info.get("battle_won", False))

            self._episode_ts += 1

            if not test_mode:
                self._env_ts += 1

        # update at timestep t_max + 1
        termination_data = {
            self._data_attr._STATE.value: [self._env.get_state()],
            self._data_attr._AVAIL_ACTIONS.value: [self._env.get_avail_actions()],
            self._data_attr._OBS.value: [self._env.get_obs()],
        }

        memory_shard.update(termination_data, time_slice=self._episode_ts)

        # Repeat conditional check for post-termination data
        if test_mode:
            actions = self._cortex.infer_greedy_actions(
                data=memory_shard,
                rollout_timestep=self._episode_ts,
                env_timestep=-1,
            )
        else:
            actions = self._cortex.infer_eps_greedy_actions(
                data=memory_shard,
                rollout_timestep=self._episode_ts,
                env_timestep=self._env_ts,
            )

        post_termination_data = {
            self._data_attr._ACTIONS.value: actions,
        }

        memory_shard.update(post_termination_data, time_slice=self._episode_ts)

        metrics["evaluation_score"] = episode_return
        metrics["evaluation_battle_won"] = battle_won

        return memory_shard, metrics

    def update_cortex_object(self, cortex) -> None:
        """update cortex instance"""
        self._cortex = cortex
