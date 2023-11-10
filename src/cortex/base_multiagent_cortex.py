from typing import List

import torch
from omegaconf import OmegaConf

from src.heuristic.policy import EpsilonGreedy
from src.learner import RecurrentQLearner
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
        self._online_drqn_network = None
        self._target_drqn_network = None

    def _ensemble_learners(
        self, n_agents: int, impl: torch.nn.Module, learner_conf: OmegaConf
    ) -> List[RecurrentQLearner]:
        """create namedtuple with N agents"""
        agents = methods.ensemble_learners(n_agents, RecurrentQLearner, learner_conf)
        return agents

    def ensemble_cortex(self, n_agents: int) -> None:
        """create heuristic, networks and ensemble N learners"""
        eps_start = self._exp_conf.epsilon_start
        eps_min = self._exp_conf.epsilon_min
        eps_anneal_time = self._exp_conf.epsilon_anneal_steps

        self._agents = self._ensemble_learners(
            n_agents, RecurrentQLearner, self._model_conf
        )
        self._policy = EpsilonGreedy(eps_start, eps_min, eps_anneal_time)

    def compute_actions(
        self,
        observations: torch.Tensor,
        avail_actions: torch.Tensor,
        timestep: int,
        evaluate: bool,
    ) -> torch.Tensor:
        pass

    def parameters(self) -> torch.Tensor:
        """return agents optimization params"""
        pass

    def rnd_seed(self, *, seed: int = None):
        """set random generator seed"""
        pass
