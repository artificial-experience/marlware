import random

import numpy as np
import torch
from omegaconf import OmegaConf

from src.cortex import MultiAgentCortex
from src.environ.starcraft import SC2Environ
from src.memory.buffer import GenericReplayMemory
from src.memory.buffer import initialize_memory
from src.registry import global_registry
from src.trainable import arch


class TrainableConstruct:
    """
    Abstraction class meant to delegate certain construct for optimization
    Based on conf and registry the construct is instantiated

    Args:
        :param [conf]: construct configuration OmegaConf

    Internal State:
        :param [trainable]: chosen trainable component to be used (e.g. BaseQMIX)
        :param [multi_agent_cortex]: controller to be used for multi-agent scenario
        :param [memory]: replay memory instance
        :param [environ]: environment instance
        :param [environ_info]: environment informations (e.g. number of actions)

    """

    def __init__(self, conf: OmegaConf) -> None:
        self._conf = conf

        # internal attrs
        self._trainable: arch.TrainableComponent = None
        self._mac = None
        self._memory = None
        self._environ = None
        self._environ_info = None

        # optimization attrs
        self._criterion = None
        self._optimizer = None

    def _integrate_trainable(
        self, construct: str, env_info: dict, seed: int
    ) -> arch.TrainableComponent:
        """check for registered constructs and integrate chosen one"""
        registered_constructs = global_registry.get_registered()
        is_registered = construct in registered_constructs
        assert is_registered is True, f"Construct {construct} is not registered"

        construct_hypernet_conf = self._conf.trainable.hypernetwork
        construct_mixer_conf = self._conf.trainable.mixer

        trainable = global_registry.get(construct)(
            construct_hypernet_conf, construct_mixer_conf
        )
        n_agents = env_info.get("n_agents", None)
        obs_dim = env_info.get("obs_shape", None)
        state_dim = env_info.get("state_shape", None)
        trainable.ensemble_construct(
            n_agents=n_agents, observation_dim=obs_dim, state_dim=state_dim, seed=seed
        )
        return trainable

    def _integrate_multi_agent_cortex(
        self, model_conf: OmegaConf, exp_conf: OmegaConf, env_info: dict, seed: int
    ) -> MultiAgentCortex:
        """create multi-agent cortex for N agents"""
        n_agents = env_info.get("n_agents", None)
        n_actions = env_info.get("n_actions", None)
        obs_dim = env_info.get("obs_shape", None)
        mac = MultiAgentCortex(model_conf, exp_conf)
        mac.ensemble_cortex(n_agents, n_actions, obs_dim, seed=seed)
        return mac

    def _integrate_memory(
        self, memory_conf: OmegaConf, env_info: dict
    ) -> GenericReplayMemory:
        """create instance of replay memory based on environ info and memory conf"""
        state_shape = env_info.get("state_shape", None)
        obs_shape = env_info.get("obs_shape", None)
        n_actions = env_info.get("n_actions", None)
        n_agents = env_info.get("n_actions", None)

        max_size = memory_conf.max_size
        batch_size = memory_conf.batch_size
        prioritized = memory_conf.prioritized

        prev_actions_field = "prev_actions"
        prev_actions_vals = np.zeros([max_size, n_agents], dtype=np.int64)

        avail_actions_field = "avail_actions"
        avail_actions_vals = np.zeros([max_size, n_agents, n_actions], dtype=np.int64)

        states_field = "states"
        states_vals = np.zeros([max_size, state_shape], dtype=np.float32)

        next_states_field = "next_states"
        next_states_vals = np.zeros([max_size, state_shape], dtype=np.float32)

        extra_fields = (
            prev_actions_field,
            states_field,
            next_states_field,
            avail_actions_field,
        )
        extra_vals = (
            prev_actions_vals,
            states_vals,
            next_states_vals,
            avail_actions_vals,
        )

        memory = initialize_memory(
            obs_shape=(obs_shape,),
            n_actions=n_actions,
            n_agents=n_agents,
            max_size=max_size,
            batch_size=batch_size,
            prioritized=prioritized,
            extra_fields=extra_fields,
            extra_vals=extra_vals,
        )
        return memory

    def _integrate_environ(self, map_name: str) -> SC2Environ:
        """based on map_name create sc2 environ instance"""
        env_manager = SC2Environ(map_name)
        env, env_info = env_manager.create_env_instance()
        assert env is not None, "Environment cound not be created"
        return env, env_info

    def _rnd_seed(self, *, seed: int = None):
        """set random seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def commit(
        self, environ_prefix: str, accelerator: str, *, seed: int = None
    ) -> None:
        """based on conf delegate construct object with given parameters"""
        self._rnd_seed(seed=seed)

        # ---- ---- ---- ---- ---- #
        # --- Integrate Environ -- #
        # ---- ---- ---- ---- ---- #

        self._environ, self._environ_info = self._integrate_environ(environ_prefix)

        # ---- ---- ---- ---- ---- #
        # -- Integrate Construct - #
        # ---- ---- ---- ---- ---- #

        construct: str = self._conf.trainable.construct.impl
        self._trainable: arch.TrainableComponent = self._integrate_trainable(
            construct, self._environ_info, seed
        )

        # ---- ---- ---- ---- ---- #
        # --- Integrate Memory --- #
        # ---- ---- ---- ---- ---- #

        memory_conf = self._conf.buffer
        self._memory = self._integrate_memory(memory_conf, self._environ_info)

        # ---- ---- ---- ---- ---- #
        # --- Integrate Cortex --- #
        # ---- ---- ---- ---- ---- #

        model_conf = self._conf.learner.model
        exp_conf = self._conf.learner.exploration
        self._mac = self._integrate_multi_agent_cortex(
            model_conf, exp_conf, self._environ_info, seed
        )

    def optimize(self, n_rollouts: int = 100) -> np.ndarray:
        """optimize construct within n_rollouts"""
        pass

    def evaluate(self, n_games: int = 10) -> np.ndarray:
        """evaluate construct on n_games"""
        pass

    def save_models(self) -> bool:
        """save all models"""
        pass

    def load_models(self, path_to_models: str) -> bool:
        """load all models given path"""
        pass
