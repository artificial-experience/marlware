from smac.env import StarCraft2Env

from qmix.common import methods


class SC2Environment:
    def __init__(self, config: dict) -> None:
        self._config = config
        self._env = None

    def create_env_instance(self):
        map_name = methods.get_nested_dict_field(
            directive=self._config,
            keys=["prefix", "choice"],
        )
        env = StarCraft2Env(map_name=map_name)

        env_info = env.get_env_info()
        n_actions = env_info["n_actions"]
        n_agents = env_info["n_agents"]
        info = {"n_actions": n_actions, "n_agents": n_agents}

        return env, info
