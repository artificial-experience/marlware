import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DRQN(nn.Module):
    """
    Deep recurrent Q-network implementation with GRUCell
    Network's forward method operates on obesrvation and previous action which is a part of observation vector
    Network will return approximated q-values and updated cell state and hidden state

    Args:
        :param [rnn_hidden_dim]: shape of hidden state array

    """

    def __init__(self, rnn_hidden_dim: int) -> None:
        super().__init__()
        self._rnn_hidden_dim = rnn_hidden_dim

    def _rnd_seed(self, *, seed: Optional[int] = None):
        """set random generator seed"""
        if seed:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def integrate_network(
        self, input_dim: int, n_q_values: int, *, seed: Optional[int] = None
    ):
        """given input dimension construct network"""
        self._rnd_seed(seed=seed)

        self._fc1 = nn.Linear(input_dim, self._rnn_hidden_dim)
        self._rnn = nn.GRUCell(self._rnn_hidden_dim, self._rnn_hidden_dim)
        self._fc2 = nn.Linear(self._rnn_hidden_dim, n_q_values)

    def init_hidden_state(self, batch_size: int) -> torch.Tensor:
        """return initial hidden state tensor filled with 0s that are on the same device as model"""
        return self._fc1.weight.new(batch_size, self._rnn_hidden_dim).zero_()

    def forward(
        self,
        feed: torch.Tensor,
        hidden: torch.Tensor = None,
    ):
        # batch_size X embedding - e.g. torch.tensor([ 32, 102 ])
        bs, embed = feed.size()

        # first layer inference and relu activation
        out = F.relu(self._fc1(feed), inplace=True)

        # reshape hidden state in case it does not match embedding dimension
        if hidden is not None:
            hidden = hidden.reshape(-1, self._rnn_hidden_dim)

        # rnn feed forward
        updated_hidden = self._rnn(out, hidden)

        # output layer q values computation
        q_vals = self._fc2(updated_hidden)

        return q_vals.view(bs, -1), updated_hidden.view(bs, -1)
