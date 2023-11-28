from typing import Optional

from omegaconf import OmegaConf
from smac.env import StarCraft2Env

from src.cortex import MultiAgentCortex
from src.memory.replay import GenericMemoryReplay


class EpisodicWorker:
    """Interface class between environment and memory replay

    Args:
        :param [conf]: worker configuration omegaconf

    Internal state:
        :param [env]: existing environment instance to be used for interaction
        :param [cortex]: existing multi-agent cortex which is used for action calculation
        :param [mem_replay]: instance of memory replay used for storing trajectories
    """

    def __init__(self, conf: OmegaConf) -> None:
        self._conf = conf

        # Internal params
        self._env = None
        self._cortex = None
        self._mem_replay = None

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def ensemble_memory_worker(
        self,
        env: StarCraft2Env,
        cortex: MultiAgentCortex,
        *,
        device: Optional[str] = "cpu",
        seed: Optiona[int] = None
    ) -> None:
        """ensemble interaction worker"""
        self._rnd_seed(seed=seed)

        self._env = env
        self._cortex = cortex

        # fetch from conf
        scheme = None
        groups = None
        mam_size = None
        max_ep_length = None
        transforms = None
