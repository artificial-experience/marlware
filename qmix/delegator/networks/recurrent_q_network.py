import numpy as np
import torch as T
import torch.functional as F
import torch.nn as nn
import torch.optim as optim

from qmix.common import methods


class DRQN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._model = None

        # Architecture configuration
        self._model_input_size = None
        self._model_mlp_before_rnn = None
        self._model_rnn_hidden_state = None
        self._model_rnn_stack = None
        self._model_n_q_values = None

        self._training_lr = None
        self._training_gamma = None
        self._training_epsilon_max = None
        self._training_epsilon_min = None
        self._training_num_epsilon_dec_steps = None
        self._target_network_update_schedule = None

    def _access_config_params(self):
        # Model configuration
        self._model_input_size = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "input_size"],
        )
        self._model_mlp_before_rnn = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "mlp_before_rnn"],
        )
        self._model_rnn_hidden_state = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "rnn_hidden_state"],
        )
        self._model_rnn_stack = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "rnn_stack"],
        )
        self._model_n_q_values = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "n_q_values"],
        )

        # Training configuration
        self._training_lr = methods.get_nested_dict_field(
            directive=self._config,
            keys=["training", "lr", "choice"],
        )
        self._training_gamma = methods.get_nested_dict_field(
            directive=self._config,
            keys=["training", "gamma", "choice"],
        )
        self._training_epsilon_max = methods.get_nested_dict_field(
            directive=self._config,
            keys=["training", "epsilon_max", "choice"],
        )
        self._training_epsilon_min = methods.get_nested_dict_field(
            directive=self._config,
            keys=["training", "epsilon_min", "choice"],
        )
        self._training_num_epsilon_dec_steps = methods.get_nested_dict_field(
            directive=self._config,
            keys=["training", "num_epsilon_dec_steps", "choice"],
        )
        self._training_target_network_update_schedule = methods.get_nested_dict_field(
            directive=self._config,
            keys=["training", "target_network_update_schedule", "choice"],
        )

    def construct_network(self):
        self._access_config_params()
        self._model = nn.Sequential(
            nn.Linear(self._model_input_size, self._model_mlp_before_rnn),
            nn.GRU(
                self._model_mlp_before_rnn,
                self._model_rnn_hidden_state,
                self._model_rnn_stack,
            ),
            nn.Linear(self._model_rnn_hidden_state, self._model_n_q_values),
        )

    def forward(self, observation: tuple):
        pass
