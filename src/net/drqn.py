import random

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class DRQN(nn.Module):
    """
    Deep recurrent Q-network implementation with GRUCell
    Network's forward method operates on obesrvation and previous action
    Network will return approximated q-values and updated cell state and hidden state

    Args:
        :param [rnn_hidden_dim]: shape of hidden state array

    """

    def __init__(self, rnn_hidden_dim: int) -> None:
        super().__init__()
        self._rnn_hidden_dim = rnn_hidden_dim

    def _rnd_seed(self, *, seed: int = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def integrate_network(self, input_dim: int, n_q_values: int, *, seed: int = None):
        """given input dimension construct network"""
        self._rnd_seed(seed=seed)

        # Initial MLP: (observation + last action one hot encoded + agent id one hot encoded) -> embedding
        self._fc1 = nn.Sequential(
            nn.Linear(input_dim, self._rnn_hidden_dim),
            nn.ReLU(inplace=True),
        )

        self._rnn = nn.GRUCell(self._rnn_hidden_dim, self._rnn_hidden_dim)

        self._fc2 = nn.Linear(self._rnn_hidden_dim, n_q_values)

    def init_hidden_state(self) -> torch.Tensor:
        """return initial hidden state tensor filled with 0s that are on the same device as model"""
        return self._fc1.weight.new(1, self._rnn_hidden_dim).zero_()

    def __call__(
        self,
        feed: torch.Tensor,
        hidden_state: torch.Tensor = None,
    ):
        # batch_size X n_agents X embedding - [ 32, 8, 102 ]
        bs, n_agents, embed = feed.size()

        out = self._fc1(feed)

        # reshape hidden state in case it does not match embedding dimension
        if hidden is not None:
            hidden = hidden.reshape(-1, self._rnn_hidden_dim)

        hidden = self._rnn(out, hidden_state)
        q_vals = self._fc2(hidden)

        return q_vals.view(bs, n_agents, -1), hidden.view(bs, n_agents, -1)
