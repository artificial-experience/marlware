import torch
import torch.nn as nn

from qmix.common import methods


class DRQN(nn.Module):
    """
    Deep recurrent Q-network implementation
    Network's forward method operates on obesrvation and previous action
    Network will return approximated q-values and updated cell state and hidden state

    Args:
        :param [config]: network configuration dictionary

    Internal State:
        :param [model_observation_size]: shape of observation array
        :param [model_n_actions]: number of actions available to approximate q-values
        :param [model_embedding_size]: shape of embedding array
        :param [model_hidden_state_size]: shape of hidden state array
        :param [model_n_q_values]: number of q-values to approximate (output from network)

    """

    def __init__(self, config: dict):
        super().__init__()
        self._config = config

        # Architecture configuration
        self._model_observation_size = None
        self._model_n_actions = None
        self._model_embedding_size = None
        self._model_hidden_state_size = None
        self._model_n_q_values = None

    def _access_config_params(self):
        """extract values given config dict"""
        self._model_observation_size = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "observation_size"],
        )
        self._model_n_actions = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "n_actions"],
        )
        self._model_embedding_size = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "embedding_size"],
        )
        self._model_hidden_state_size = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "hidden_state_size"],
        )
        self._model_n_q_values = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "n_q_values"],
        )

    @property
    def hidden_state_dim(self):
        """getter methd for hidden state size"""
        return self._model_hidden_state_size

    def _init_weights(self, x):
        """weight initializer method - xavier"""
        if type(x) == nn.Linear:
            nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)

    def construct_network(self, num_agents: int):
        """given number of agents the method will construct each agent and return itself"""
        self._access_config_params()

        # Initial MLP: (observation + last action one hot encoded) -> embedding
        self.mlp1 = nn.Sequential(
            nn.Linear(
                self._model_observation_size + self._model_n_actions + num_agents,
                self._model_embedding_size,
            ),
            nn.ReLU(),
        )

        self.lstm = nn.LSTMCell(
            self._model_embedding_size, self._model_hidden_state_size
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(self._model_hidden_state_size, self._model_n_q_values)
        )

        # Apply Xavier initialisation by recursive search
        self.apply(self._init_weights)

        return self

    def forward(
        self,
        observation: torch.Tensor,
        prev_action: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
    ):
        """network inference method"""
        # Ensure that the tensors are 3-dimensional (batch size can be 1)
        if len(observation.size()) < 3:
            observation = observation.unsqueeze(0)
        if len(prev_action.size()) < 3:
            prev_action = prev_action.unsqueeze(0)

        collective_input = torch.cat([observation, prev_action], axis=-1)
        x = self.mlp1(collective_input)
        x = x.squeeze(0)

        updated_hidden_states = []
        for batch_idx in range(x.size(0)):
            updated_hidden_state, updated_cell_state = self.lstm(
                x[batch_idx, :], (hidden_state, cell_state)
            )
            updated_hidden_states.append(updated_hidden_state)

        t_updated_hidden_states = torch.stack(updated_hidden_states, 0)

        q_values = self.mlp2(t_updated_hidden_states)
        return q_values, (updated_hidden_state, updated_cell_state)
