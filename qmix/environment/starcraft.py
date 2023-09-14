from smac.env import StarCraft2Env


class SC2Environment:
    def __init__(self, config: dict) -> None:
        self._config = config
        self._env = None

    def _access_env_configuration(self):
        map_name = self._config.get("env_name")
        self._env = StarCraft2Env(map_name=map_name)

    def _get_map_informations(self):
        self._access_env_configuration()
        env_info = self._env.get_env_info()

        n_actions = env_info["n_actions"]
        n_agents = env_info["n_agents"]
        return n_actions, n_agents


if __name__ == "__main__":
    config = {"env_name": "8m"}
    env = SC2Environment(config)
    x = env._get_map_informations()
