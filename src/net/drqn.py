import torch
import torch.nn as nn
from omegaconf import OmegaConf


class DRQN(nn.Module):
    """
    Deep recurrent Q-network implementation with GRUCell
    Network's forward method operates on obesrvation and previous action
    Network will return approximated q-values and updated cell state and hidden state

    Args:
        :param [conf]: network configuration OmegaConf

    Internal State:
        :param [model_embedding_dim]: shape of embedding array
        :param [model_hidden_state_dim]: shape of hidden state array

    """

    def __init__(self, conf: OmegaConf):
        super().__init__()
        self._conf = conf

        self._embedding_dim = None
        self._hidden_state_dim = None

    def integrate_network(self, n_agents: int, n_actions: int, observation_dim: tuple):
        """given number of agents the method will construct each agent and return itself"""

        # Initial MLP: (observation + last action one hot encoded + agent id one hot encoded) -> embedding
        self._mlp1 = nn.Sequential(
            nn.Linear(
                self._model_observation_size + self._model_n_actions + n_agents,
                self._model_embedding_size,
            ),
            nn.ReLU(),
        )

        self._gru = nn.GRUCell(
            self._model_embedding_size, self._model_hidden_state_size
        )

        self._mlp2 = nn.Sequential(
            nn.Linear(self._model_hidden_state_size, self._model_n_q_values)
        )

    def __call__(
        self,
        observation: torch.Tensor,
        prev_action: torch.Tensor,
        hidden_state: torch.Tensor,
    ):
        """network inference method"""
        collective_input = torch.cat([observation, prev_action], axis=-1)
        x = self._mlp1(collective_input)
        x = x.squeeze(0)

        updated_hidden_states = []
        for batch_idx in range(x.size(0)):
            updated_hidden_state = self._gru(x[batch_idx, :], hidden_state)
            updated_hidden_states.append(updated_hidden_state)

        t_updated_hidden_states = torch.stack(updated_hidden_states, 0)

        q_values = self._mlp2(t_updated_hidden_states)
        return q_values, updated_hidden_state
