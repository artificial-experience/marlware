import numpy as np
import torch as T
import torch.functional as F
import torch.nn as nn
import torch.optim as optim

from qmix.common import methods


class MixingNetwork(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._model = None

    def _access_config_params(self):
        # Model configuration
        self._model_input_size = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "input_size"],
        )

    def construct_network(self):
        pass

    def forward(self, observation: tuple):
        pass
