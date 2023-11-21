import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    """
    Mixing Network implementation for value based training
    Takes as input agent q values and calculates single value
    Allows for decomposition and credit assignment for agents
    Uses hypernetworks to create weights for main net
    Uses abs activation to ensure monotonicity of updates

    Args:
        :param [hypernet_embed_dim]: dimension of hypernetwork embedding
        :param [mixer_embed_dim]: dimension of mixer head embedding
        :param [n_hypernet_layers]: number of hypernet layers to calculate weights

    Internal State:
        :pram [state_dim]: dimension of information for centraliazed training
        :pram [n_agents]: number of learners

    """

    def __init__(
        self, hypernet_embed_dim: int, mixer_embed_dim: int, n_hypernet_layers: int
    ) -> None:
        super().__init__()
        self._hypernet_embed_dim = hypernet_embed_dim
        self._mixer_embed_dim = mixer_embed_dim
        self._n_hypernet_layers = n_hypernet_layers

        # internal attrs
        self._n_agents = None
        self._state_dim = None

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def integrate_network(
        self, n_agents: int, state_dim: int, *, seed: Optional[int] = None
    ):
        """given number of agents the method will construct each agent and return itself"""
        self._rnd_seed(seed=seed)

        # keep info for mixer magic
        self._n_agents = n_agents
        self._state_dim = state_dim

        # State-dependent weights for the first layer
        self._hyper_w_1 = nn.Sequential(
            nn.Linear(self._state_dim, self._hypernet_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self._hypernet_embed_dim, self._mixer_embed_dim * self._n_agents),
        )

        # State-dependent bias for the first layer
        self._hyper_b_1 = nn.Linear(self._state_dim, self._mixer_embed_dim)

        # State-dependent weights for the second layer
        self._hyper_w_final = nn.Sequential(
            nn.Linear(self._state_dim, self._hypernet_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self._hypernet_embed_dim, self._mixer_embed_dim),
        )

        # V(s) as a bias for the final output
        self._V = nn.Sequential(
            nn.Linear(self._state_dim, self._mixer_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self._mixer_embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor):
        # state shape: [batch_size, 168]
        states = states.reshape(-1, self._state_dim)

        # First Layer
        w1 = torch.abs(self._hyper_w_1(states))
        b1 = self._hyper_b_1(states)

        # w1 shape: [batch_size, 8, 32]
        w1 = w1.reshape(-1, self._n_agents, self._mixer_embed_dim)
        # b1 shape: [batch_size, 1, 32]
        b1 = b1.reshape(-1, 1, self._mixer_embed_dim)
        # agent_qs shape: [batch_size, 1, 8]
        agent_qs = agent_qs.reshape(-1, 1, self._n_agents)

        # hidden shape: [batch_size, 1, 32]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w_final = torch.abs(self._hyper_w_final(states))

        # w_final shape: [batch_size, 32, 1]
        w_final = w_final.reshape(-1, self._mixer_embed_dim, 1)

        value = self._V(states)
        # value shape: [batch_size, 1, 1]
        value = value.reshape(-1, 1, 1)

        # q_tot shape: [batch_size, 1, 1]
        q_tot = torch.bmm(hidden, w_final) + value

        return q_tot
