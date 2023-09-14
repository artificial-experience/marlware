import torch
import torch.nn as nn

from qmix.agent.networks import HyperNetworkForBiases
from qmix.agent.networks import HyperNetworkForWeights
from qmix.common import methods


class MixingNetwork(nn.Module):
    def __init__(
        self, mixing_network_configuration: dict, hypernetwork_configuration: dict
    ):
        super().__init__()
        self._mixing_network_configuration = mixing_network_configuration
        self._hypernetwork_configuration = hypernetwork_configuration

    def _access_config_params(self):
        # Model configuration
        self._model_input_size = methods.get_nested_dict_field(
            directive=self._config,
            keys=["model", "choice", "input_size"],
        )

    def _init_weights(self, x):
        if type(x) == nn.Linear:
            nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)

    def construct_network(self):
        # Apply Xavier initialisation by recursive search
        self.apply(self._init_weights)

    def forward(self, observation: torch.Tensor):
        pass
