import numpy as np
import torch as T
import torch.functional as F
import torch.nn as nn
import torch.optim as optim

from qmix.common import methods


class HyperNetwork(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._model = None

        # Architecture configuration
        self._model_input_size = None
        self._model_mlp = None
        self._model_target_network_weights_size = None

    def _access_config_params(self):
        # Model configuration
        self._model_input_size = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "input_size"],
        )
        self._model_mlp = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "mlp"],
        )
        self._model_host_network_param_dims = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "host_network_param_dims"],
        )

    def construct_network(self):
        self._access_config_params()

    def forward(self, observation: tuple):
        pass
