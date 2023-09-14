import torch
import torch.nn as nn

from qmix.common import methods


class DRQN(nn.Module):
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

    def _init_weights(self, x):
        if type(x) == nn.Linear:
            nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)

    def construct_network(self):
        self._access_config_params()

        # Initial MLP: (observation + last action one hot encoded) -> embedding
        self.mlp1 = nn.Sequential(
            nn.Linear(
                self._model_observation_size + self._model_n_actions,
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
        joint_input = torch.cat([observation, prev_action], axis=1)
        x = self.mlp1(joint_input)
        updated_hidden_state, updated_cell_state = self.rnn(
            x, (hidden_state, cell_state)
        )
        q_values = self.mlp2(updated_hidden_state)
        return q_values, (updated_hidden_state, updated_cell_state)
