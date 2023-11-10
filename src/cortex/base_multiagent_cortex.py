import copy
import random
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

from src.heuristic.policy import EpsilonGreedy
from src.learner import RecurrentQLearner
from src.net import DRQN
from src.util import methods


class MultiAgentCortex:
    """
    Interface for multi-agent coordination and optimization [ mac ]
    NOTE: Shared parameters implementation

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

    def _rnd_seed(self, *, seed: int = None):
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
        self, n_agents: int, n_actions: int, obs_shape: tuple, *, seed: int = None
    ) -> None:
        """create heuristic, networks and ensemble N learners"""
        self._rnd_seed(seed=seed)

        # ---- ---- ---- ---- ---- ---- #
        # --- Prepare Heuristic --- --- #
        # ---- ---- ---- ---- ---- ---- #

        eps_start = self._exp_conf.epsilon_start
        eps_min = self._exp_conf.epsilon_min
        eps_anneal_time = self._exp_conf.epsilon_anneal_steps

        self._policy = EpsilonGreedy(eps_start, eps_min, eps_anneal_time)

        # ---- ---- ---- ---- ---- ---- #
        # ---- Prepare Networks --- --- #
        # ---- ---- ---- ---- ---- ---- #

        rnn_hidden_dim = self._model_conf.rnn_hidden_dim
        self._eval_drqn_network = DRQN(rnn_hidden_dim)

        # dim = agent one hot id + num q values + obs shape
        input_dim = n_agents + n_actions + obs_shape
        # the same number of q_values as actions
        n_q_values = n_actions
        self._eval_drqn_network.integrate_network(input_dim, n_q_values, seed=seed)

        # deepcopy eval network structure for frozen target net
        self._target_drqn_network = copy.deepcopy(self._eval_drqn_network)

        # ---- ---- ---- ---- ---- ---- #
        # ---- Prepare Learners --- --- #
        # ---- ---- ---- ---- ---- ---- #

        self._agents = self._ensemble_learners(
            n_agents, RecurrentQLearner, self._model_conf
        )

        # share network between agents
        for agent in self._agents:
            agent.share_networks(
                eval_net=self._eval_drqn_network, target_net=self._target_drqn_network
            )

    def compute_actions(
        self,
        observations: torch.Tensor,
        avail_actions: torch.Tensor,
        timestep: int,
        evaluate: bool,
    ) -> torch.Tensor:
        pass

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

    def parameters(self) -> torch.Tensor:
        """return eval net optimization params"""
        return self._eval_drqn_network.parameters()

    def move_to_cuda(self):
        """move models to cuda device"""
        self._eval_drqn_network.cuda()
        self._target_drqn_network.cuda()
