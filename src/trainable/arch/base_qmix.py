import torch
from omegaconf import OmegaConf

from src.abstract import BaseConstruct
from src.registry import register_construct


@register_construct
class BaseQMIX(BaseConstruct):
    """
    Base implementation of QMIX: Monotonic Value Function Factorisation
    for Deep Multi-Agent Reinforcement Learning

    Call method takes torch batch and computes factorised q-value
    Expected output shape: [ BATCH_SIZE, 1, 1 ]

    Args:
        :param [hypernet_conf]: hypernetwork configuration
        :param [mixer_conf]: mixer head configuration

    """

    def __init__(self, hypernet_conf: OmegaConf, mixer_conf: OmegaConf) -> None:
        self._hypernet_conf = hypernet_conf
        self._mixer_conf = mixer_conf

    def _rnd_seed(self, *, seed: int = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def ensemble_construct(
        self,
        n_agents: int,
        n_actions: int,
        observation_dim: tuple,
        state_dim: tuple,
        *,
        seed: int = None
    ) -> None:
        self._rnd_seed(seed=seed)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """takes batch and computes factorised q-value"""
        pass

    def parameters(self):
        """return hypernet and mixer optimization params"""
        pass

    def rnd_seed(self, *, seed: int = None):
        """set random generator seed"""
        pass
