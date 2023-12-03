import copy
import random
from functools import partialmethod
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.heuristic.policy import EpsilonGreedy
from src.learner import RecurrentQLearner
from src.net import DRQN
from src.util import methods
from src.util.constants import AttrKey


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
        seed: Optional[int] = None
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

    def infer_actions(
        self,
        data: dict,
        rollout_timestep: int,
        env_timestep: int,
        evaluate: bool = False,
    ) -> np.ndarray:
        """get feed and compute agents actions to take in the environment"""

        data_attr = AttrKey.data

        # prepare feed and expand tensor to 4d
        observations = data[data_attr._OBS.value][:, rollout_timestep].unsqueeze(0)
        avail_actions = data[data_attr._AVAIL_ACTIONS.value][
            :, rollout_timestep
        ].unsqueeze(0)
        actions = data[data_attr._ACTIONS.value][:, rollout_timestep].unsqueeze(0)

        # prepare fed for the agent
        feed = {
            data_attr._OBS.value: observations,
            data_attr._AVAIL_ACTIONS.value: avail_actions,
            data_attr._ACTIONS.value: actions,
        }

        bs, time, n_agents, n_q_values = avail_actions.shape

        t_multi_agent_q_vals = self.estimate_eval_q_vals(feed)
        t_multi_agent_q_vals = t_multi_agent_q_vals.detach()

        # n_agents X batch_size X n_q_vals
        avail_actions = avail_actions.view(-1, n_agents, n_q_values)

        decided_actions = (
            self._policy.decide_actions_greedily(
                t_multi_agent_q_vals, avail_actions, env_timestep
            )
            if evaluate
            else self._policy.decide_actions_epsilon_greedily(
                t_multi_agent_q_vals, avail_actions, env_timestep
            )
        )
        return decided_actions.detach().numpy()

    def estimate_q_vals(
        self, feed: Dict[str, torch.Tensor], use_target: bool = False
    ) -> torch.Tensor:
        """either use eval or target net to estimate q values"""
        multi_agent_q_vals = []
        for agent in self._agents:
            q_vals = (
                agent.estimate_target_q(feed)
                if use_target
                else agent.estimate_eval_q(feed)
            )

            multi_agent_q_vals.append(q_vals)

        # n_agents X batch_size X n_q_vals
        t_multi_agent_q_vals = torch.stack(multi_agent_q_vals, dim=1)
        return t_multi_agent_q_vals

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

    def init_hidden(self, batch_size: int) -> None:
        """set internal hidden state parameters to zero"""
        self._eval_drqn_network.init_hidden_state(batch_size=batch_size)
        self._target_drqn_network.init_hidden_state(batch_size=batch_size)

    # ---- ---- ---- ---- ---- #
    # @ -> Partial Methods
    # ---- ---- ---- ---- ---- #

    infer_eps_greedy_actions = partialmethod(infer_actions, evaluate=False)
    infer_greedy_actions = partialmethod(infer_actions, evaluate=True)

    estimate_eval_q_vals = partialmethod(estimate_q_vals, use_target=False)
    estimate_target_q_vals = partialmethod(estimate_q_vals, use_target=True)
