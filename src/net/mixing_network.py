import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


class MixingNetwork(nn.Module):
    """
    Deep recurrent Q-network implementation
    Network's forward method operates on obesrvation and previous action
    Network will return approximated q-values and updated cell state and hidden state

    Args:
        :param [conf]: network configuration OmegaConf

    Internal State:
        :param [model_observation_size]: shape of observation array
        :param [model_n_actions]: number of actions available to approximate q-values
        :param [model_embedding_size]: shape of embedding array
        :param [model_hidden_state_size]: shape of hidden state array
        :param [model_n_q_values]: number of q-values to approximate (output from network)

    """

    def __init__(self, conf: OmegaConf) -> None:
        super().__init__()
        self._conf = conf

        self._hypernet_embed_dim = None
        self._mixer_embed_dim = None

    def integrate_network(self, n_agents: int, state_dim: int):
        """given number of agents the method will construct each agent and return itself"""

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

    def forward(self, agent_qs, states):
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
