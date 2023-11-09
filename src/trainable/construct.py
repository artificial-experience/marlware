import numpy as np
from omegaconf import OmegaConf

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
        :param [multi_agent_controller]: controller to be used for multi-agent scenario
        :param [memory]: replay memory instance
        :param [environ]: environment instance
        :param [environ_info]: environment informations (e.g. number of actions)

    """

    def __init__(self, conf: OmegaConf) -> None:
        self._conf = conf

        self._trainable: arch.TrainableComponent = None
        self._multi_agent_controller = None
        self._memory = None
        self._environ = None
        self._environ_info = None

    def _integrate_trainable(self, construct: str) -> arch.TrainableComponent:
        """check for registered constructs and integrate chosen one"""
        registered_constructs = global_registry.get_registered()
        is_registered = construct in registered_constructs
        assert is_registered is True, f"Construct {construct} is not registered"

        construct_hypernet_conf = self._conf.trainable.hypernetwork
        construct_mixer_conf = self._conf.trainable.mixer

        trainable = global_registry.get(construct)(
            construct_hypernet_conf, construct_mixer_conf
        )
        return trainable

    def _integrate_multi_agent_controller(self):
        pass

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

    def commit(self, environ_prefix: str, accelerator: str, *, seed=None) -> None:
        """based on conf delegate construct object with given parameters"""
        construct: str = self._conf.trainable.construct.impl
        self._trainable: arch.TrainableComponent = self._integrate_trainable(construct)

        self._environ, self._environ_info = self._integrate_environ(environ_prefix)

        memory_conf = self._conf.buffer
        self._memory = self._integrate_memory(memory_conf, self._environ_info)

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
