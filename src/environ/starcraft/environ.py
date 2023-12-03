from typing import Dict
from typing import Tuple

from smac.env import StarCraft2Env


# TODO: Refactor this code - its shit
class SC2Environ:
    """
    Abstraction layer for SC2 environment handler

    Args:
        :param [map_name]: map name for the sc2 env to render

    """

    def __init__(self, map_name: str) -> None:
        self._map_name = map_name

    def create_env_instance(self) -> Tuple[StarCraft2Env, Dict]:
        """create sc2 environ based on passed map name and return along with info"""
        env = StarCraft2Env(map_name=self._map_name)
        env_info = env.get_env_info()
        return env, env_info
