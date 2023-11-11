from typing import Dict
from typing import Tuple

import numpy as np
from omegaconf import OmegaConf

from src.cortex import MultiAgentCortex
from src.environ.starcraft import SC2Environ
from src.memory.buffer import GenericReplayMemory


class SynchronousCollector:
    def __init__(self, conf: OmegaConf) -> None:
        self._conf = conf

        # sampling mode
        self._sampling_mode = "uniform"

        # internal attrs
        self._memory = None
        self._environ = None
        self._environ_info = None

        # counter
        self._timesteps = 0

    def ensemble_collector(
        self,
        replay_memory: GenericReplayMemory,
        environ: SC2Environ,
        environ_info: dict,
    ) -> None:
        """store replay memory instance, environ instance and multi agent cortex"""
        sampling_mode = self._conf.mode
        self._sampling_mode = sampling_mode

        self._memory = replay_memory
        self._environ = environ
        self._environ_info = environ_info

    def _get_avail_actions(self, n_agents: int) -> list:
        """return a list with available actions for each agent"""
        avail_actions = []
        for agent_id in range(n_agents):
            available_actions = self._environ.get_avail_agent_actions(agent_id)
            avail_actions.append(available_actions)

        return np.array(avail_actions, dtype=np.int64)

    def _store_trajectory(self, feed: Dict[str, np.ndarray]) -> None:
        """store feed trajectory into replay memory"""
        # Check if any of the arrays in 'feed' are entirely zeros
        has_zero_prev_action = "prev_actions" in feed and np.all(
            feed["prev_actions"] == 0
        )

        if not has_zero_prev_action:
            # Extract data from 'feed' if no array is entirely zeros
            data = [
                feed["observations"],
                feed["actions"],
                feed["reward"],
                feed["next_observations"],
                feed["terminated"],
                feed["prev_actions"],
                feed["states"],
                feed["next_states"],
                feed["avail_actions"],
            ]
            self._memory.store_transition(data)
        else:
            # Handle the case where there is an array completely filled with zeros
            pass

    def _execute_actions(
        self, actions: np.ndarray, avail_actions: np.ndarray
    ) -> Tuple[float, bool]:
        """execute actions on env and step env"""
        avail_actions_idx = [np.nonzero(row)[0] for row in avail_actions]

        # ensure [n_agents,] shape of np array
        actions = actions.squeeze(1)
        actions_ok = all(
            [actions[i].item() in avail_actions_idx[i] for i in range(len(actions))]
        )
        assert actions_ok, "Some action was not allowed by the environment"
        reward, terminated, _ = self._environ.step(actions)
        return reward, terminated

    def roll_environ_and_collect_trajectory(self, mac: MultiAgentCortex) -> None:
        """communicate with cortex to collect trajectories and store them in replay memory"""
        self._environ.reset()
        n_agents = self._environ_info.get("n_agents", 0)
        obs_size = self._environ_info.get("obs_size", 0)
        state_size = self._environ_info.get("state_size", 0)

        terminated = False
        prev_actions = np.zeros((n_agents, 1))
        next_observations = np.zeros((n_agents, obs_size))
        next_states = np.zeros((n_agents, state_size))

        feed = {
            "observations": None,
            "actions": None,
            "reward": None,
            "next_observations": None,
            "terminated": terminated,
            "prev_actions": prev_actions,
            "states": None,
            "next_states": None,
            "avail_actions": None,
        }

        while not terminated:
            observations = np.array(self._environ.get_obs(), dtype=np.float32)
            states = np.array(self._environ.get_state(), dtype=np.float32)
            avail_actions = self._get_avail_actions(n_agents)

            actions: np.ndarray = mac.compute_eps_greedy_actions(
                observations, prev_actions, avail_actions, self._timesteps
            )
            actions = np.expand_dims(actions, axis=1)
            reward, terminated = self._execute_actions(actions, avail_actions)

            next_observations = np.array(self._environ.get_obs(), dtype=np.float32)
            next_states = np.array(self._environ.get_state(), dtype=np.float32)

            feed["observations"] = observations
            feed["actions"] = actions
            feed["reward"] = reward
            feed["next_observations"] = next_observations
            feed["terminated"] = terminated
            feed["prev_actions"] = prev_actions
            feed["states"] = states
            feed["next_states"] = next_states
            feed["avail_actions"] = avail_actions

            self._store_trajectory(feed)

            prev_actions = actions
            self._timesteps += 1

    def memory_ready(self) -> bool:
        """check if replay buffer is ready to be sampled"""
        return self._memory.ready()

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """sample data from memory and form batch"""
        # TODO: write new samping stratedy that returns adjacent episodes
        batch = self._memory.sample_buffer(mode=self._sampling_mode)
        return batch
