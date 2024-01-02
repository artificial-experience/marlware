import copy
import random
from functools import partialmethod
from pathlib import Path
from typing import Dict
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from src.abstract import ProtoTrainable
from src.net import QMixer
from src.registry import register_trainable
from src.util.constants import AttrKey


class ProtoQmix(ProtoTrainable):
    """
    Abstraction layer of QMIX: Monotonic Value Function Factorisation
    for Deep Multi-Agent Reinforcement Learning

    Args:
        :param [hypernet_conf]: hypernetwork configuration
        :param [mixer_conf]: mixer head configuration

    Internal State:
        :param [eval_mixer]: evaluation mixing network instance
        :param [target_mixer]: frozen instance of network used for target calculation

    """

    def __init__(self, hypernet_conf: OmegaConf, mixer_conf: OmegaConf) -> None:
        self._hypernet_conf = hypernet_conf
        self._mixer_conf = mixer_conf

        # internal attrs
        self._eval_mixer = None
        self._target_mixer = None

        # loss calculation
        self._gamma = None

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def ensemble_trainable(
        self,
        n_agents: int,
        observation_dim: int,
        state_dim: int,
        gamma: float,
        *,
        seed: Optional[int] = None,
    ) -> None:
        self._rnd_seed(seed=seed)

        # ---- ---- ---- ---- ---- ---- #
        # @ -> Prepare Mixers
        # ---- ---- ---- ---- ---- ---- #

        hypernet_embed_dim = self._hypernet_conf.embedding_dim
        mixer_embed_dim = self._mixer_conf.embedding_dim
        n_hypernet_layers = self._hypernet_conf.n_layers
        self._eval_mixer = QMixer(
            hypernet_embed_dim=hypernet_embed_dim,
            mixer_embed_dim=mixer_embed_dim,
            n_hypernet_layers=n_hypernet_layers,
        )
        self._eval_mixer.integrate_network(n_agents, state_dim, seed=seed)

        # deepcopy eval network structure for frozen mixer network
        self._target_mixer = copy.deepcopy(self._eval_mixer)

        # ---- ---- ---- ---- ---- ---- #
        # @ -> Prepare Hyperparams
        # ---- ---- ---- ---- ---- ---- #

        self._gamma = gamma

    def parameters(self):
        """return hypernet and mixer optimization params"""
        return self._eval_mixer.parameters()

    def move_to_device(self, device=None):
        """
        Move models to specified device.
        If no device is specified, it defaults to CUDA if available, else CPU.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._eval_mixer.to(device)
        self._target_mixer.to(device)

    def save_models(self, save_directory: Path, model_identifier: str) -> None:
        """save model weights to target directory"""
        eval_model_save_path = save_directory / f"eval_mixer_{model_identifier}"
        target_model_save_path = save_directory / "target_mixer_{}".format(
            model_identifier
        )

        torch.save(self._eval_mixer.state_dict(), eval_model_save_path)
        torch.save(self._target_mixer.state_dict(), target_model_save_path)
