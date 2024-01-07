import random
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper


class SC2Environ:
    """
    Abstraction layer for SC2 environment handler

    Args:
        :param [map_name]: map name for the sc2 env to render

    """

    def __init__(self, conf: str) -> None:
        self._conf = conf["args"]

    def create_env_instance(
        self, *, seed: Optional[int] = None
    ) -> Tuple[StarCraftCapabilityEnvWrapper, Dict]:
        """create sc2 environ based on passed environ config and return along with info"""
        seed = seed
        map_name = self._conf.get("map_name", "8m")
        continuing_episode = self._conf.get("continuing_episode", False)
        difficulty = self._conf.get("difficulty", "7")
        game_version = self._conf.get("game_version", None)
        move_amount = self._conf.get("move_amount", 2)
        obs_all_health = self._conf.get("obs_all_health", True)
        obs_instead_of_state = self._conf.get("obs_instead_of_state", False)
        obs_last_action = self._conf.get("obs_last_action", False)
        obs_own_health = self._conf.get("obs_own_health", True)
        obs_pathing_grid = self._conf.get("obs_pathing_grid", False)
        obs_terrain_height = self._conf.get("obs_terrain_height", False)
        obs_timestep_number = self._conf.get("obs_timestep_number", False)
        reward_death_value = self._conf.get("reward_death_value", 20)
        reward_defeat = self._conf.get("reward_defeat", 0)
        reward_negative_scale = self._conf.get("reward_negative_scale", 0.5)
        reward_only_positive = self._conf.get("reward_only_positive", True)
        reward_scale = self._conf.get("reward_scale", True)
        reward_scale_rate = self._conf.get("reward_scale_rate", 20)
        reward_sparse = self._conf.get("reward_sparse", False)
        reward_win = self._conf.get("reward_win", 200)
        conic_fov = self._conf.get("conic_fov", False)
        use_unit_ranges = self._conf.get("use_unit_ranges", True)
        min_attack_range = self._conf.get("min_attack_range", 2)
        obs_own_pos = self._conf.get("obs_own_pos", True)
        num_fov_actions = self._conf.get("num_fov_actions", 12)
        fully_observable = self._conf.get("fully_observable", False)

        state_last_action = self._conf.get("state_last_action", True)
        state_timestep_number = self._conf.get("state_timestep_number", False)
        step_mul = self._conf.get("step_mul", 8)
        heuristic_ai = self._conf.get("heuristic_ai", False)
        debug = self._conf.get("debug", False)
        prob_obs_enemy = self._conf.get("prob_obs_enemy", 1.0)
        action_mask = self._conf.get("action_mask", True)

        window_size_x = self._conf.get("window_size_x", 1920)
        window_size_y = self._conf.get("window_size_y", 1200)

        # smacv2 extenstion
        capability_config = self._conf.get("capability_config", None)

        env = StarCraftCapabilityEnvWrapper(
            capability_config=capability_config,
            seed=seed,
            map_name=map_name,
            debug=debug,
            conic_fov=conic_fov,
            use_unit_ranges=use_unit_ranges,
            min_attack_range=min_attack_range,
            obs_own_pos=obs_own_pos,
            fully_observable=fully_observable,
            window_size_x=window_size_x,
            window_size_y=window_size_y,
        )

        env_info = env.get_env_info()
        return env, env_info
