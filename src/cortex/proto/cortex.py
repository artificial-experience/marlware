import copy
import random
from pathlib import Path
from typing import List
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from src.abstract import ProtoCortex
from src.heuristic.policy import EpsilonGreedy
from src.learner import RecurrentQLearner
from src.net import DRQN
from src.util import methods


class ProtoCortex(ProtoCortex):
    """
    Abstraction layer for multi-agent coordination and optimization [ mac ]

    Args:
        :param [model_conf]: mac configuration OmegaConf
            contains configuration for learners
        :param [exp_conf]: mac configuration OmegaConf
            contains configuration for heuristics [ policy ]

    Internal State:
        :param [policy]: policy to be followed by each agent in mac
    """

    def __init__(self, model_conf: OmegaConf, exp_conf: OmegaConf) -> None:
        self._model_conf = model_conf
        self._exp_conf = exp_conf

        # internal attrs
        self._agents = None
        self._policy = None

        # shared attrs
        self._eval_drqn_network = None
        self._target_drqn_network = None

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def _ensemble_learners(
        self, n_agents: int, impl: torch.nn.Module, learner_conf: OmegaConf
    ) -> List[RecurrentQLearner]:
        """create namedtuple with N agents"""
        agents = methods.ensemble_learners(n_agents, RecurrentQLearner, learner_conf)
        return agents

    def ensemble_cortex(
        self,
        n_agents: int,
        n_actions: int,
        obs_shape: tuple,
        *,
        seed: Optional[int] = None,
    ) -> None:
        """create heuristic, networks and ensemble N learners"""
        self._rnd_seed(seed=seed)

        # ---- ---- ---- ---- ---- ---- #
        # @ -> Prepare Heuristic
        # ---- ---- ---- ---- ---- ---- #

        eps_start = self._exp_conf.epsilon_start
        eps_min = self._exp_conf.epsilon_min
        eps_anneal_time = self._exp_conf.epsilon_anneal_steps

        self._policy = EpsilonGreedy(eps_start, eps_min, eps_anneal_time)
        self._policy.ensemble_policy(seed=seed)

        # ---- ---- ---- ---- ---- ---- #
        # @ -> Prepare Networks
        # ---- ---- ---- ---- ---- ---- #

        rnn_hidden_dim = self._model_conf.rnn_hidden_dim
        self._eval_drqn_network = DRQN(rnn_hidden_dim)

        # dim = agent one hot id + obs shape + prev action one hot
        input_dim = n_agents + obs_shape + n_actions
        # the same number of q_values as actions
        n_q_values = n_actions
        self._eval_drqn_network.integrate_network(input_dim, n_q_values, seed=seed)

        # deepcopy eval network structure for frozen target net
        self._target_drqn_network = copy.deepcopy(self._eval_drqn_network)

        # ---- ---- ---- ---- ---- ---- #
        # @ -> Prepare Learners
        # ---- ---- ---- ---- ---- ---- #

        self._agents = self._ensemble_learners(
            n_agents, RecurrentQLearner, self._model_conf
        )

        # share network between agents
        for agent in self._agents:
            agent.share_networks(
                eval_net=self._eval_drqn_network, target_net=self._target_drqn_network
            )

    def parameters(self) -> torch.Tensor:
        """return eval net optimization params"""
        return self._eval_drqn_network.parameters()

    def move_to_device(self, device=None):
        """
        Move models to specified device.
        If no device is specified, it defaults to CUDA if available, else CPU.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._eval_drqn_network.to(device)
        self._target_drqn_network.to(device)

    def init_hidden(self, batch_size: int) -> None:
        """set internal hidden state parameters to zero"""
        self._eval_drqn_network.init_hidden_state(batch_size=batch_size)
        self._target_drqn_network.init_hidden_state(batch_size=batch_size)

    def synchronize_target_net(self, tau: float = 1.0):
        """Copy weights from eval net to target net using tau temperature.

        For tau = 1.0, this performs a hard update.
        For 0 < tau < 1.0, this performs a soft update.
        """
        for target_param, eval_param in zip(
            self._target_drqn_network.parameters(), self._eval_drqn_network.parameters()
        ):
            target_param.data.copy_(
                tau * eval_param.data + (1 - tau) * target_param.data
            )

    def save_models(self, save_directory: Path, model_identifier: str) -> None:
        """save model weights to target directory"""
        eval_model_save_path = save_directory / f"eval_net_{model_identifier}"
        target_model_save_path = save_directory / "target_net_{}".format(
            model_identifier
        )

        torch.save(self._eval_drqn_network.state_dict(), eval_model_save_path)
        torch.save(self._target_drqn_network.state_dict(), target_model_save_path)
