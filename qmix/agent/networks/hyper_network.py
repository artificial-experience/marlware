import torch
import torch.nn as nn

from qmix.common import methods


class AbsoluteActivation(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class HyperNetworkForWeights(nn.Module):
    def __init__(self, hypernetwork_configuration: dict):
        super().__init__()
        self._hypernetwork_configuration = hypernetwork_configuration

        # Architecture configuration
        self._model_state_representation_size = None
        self._model_hidden_layer_size = None

    def _access_config_params(self):
        # Model configuration
        self._model_state_representation_size = methods.get_nested_dict_field(
            directive=self._hypernetwork_configuration,
            keys=["weight_updates", "model", "choice", "state_representation_size"],
        )
        self._model_hidden_layer_size = methods.get_nested_dict_field(
            directive=self._hypernetwork_configuration,
            keys=["weight_updates", "model", "choice", "hidden_layer_size"],
        )

    def _init_weights(self, x):
        if type(x) == nn.Linear:
            nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)

    def construct_network(self, host_network_weights_size):
        self._access_config_params()
        self.mlp = nn.Sequential(
            nn.Linear(self._model_state_representation_size, host_network_weights_size),
            AbsoluteActivation(),
        )

        # Apply Xavier initialisation by recursive search
        self.apply(self._init_weights)

    def forward(self, state_representation: torch.Tensor):
        output = self.mlp(state_representation)
        return output


class HyperNetworkForBiases(nn.Module):
    def __init__(self, hypernetwork_configuration: dict):
        super().__init__()
        self._hypernetwork_configuration = hypernetwork_configuration

        # Architecture configuration
        self._model_state_representation_size = None
        self._model_hidden_layer_size = None

    def _access_config_params(self):
        # Model configuration
        self._model_state_representation_size = methods.get_nested_dict_field(
            directive=self._hypernetwork_configuration,
            keys=["bias_updates", "model", "choice", "state_representation_size"],
        )
        self._model_hidden_layer_size = methods.get_nested_dict_field(
            directive=self._hypernetwork_configuration,
            keys=["bias_updates", "model", "choice", "hidden_layer_size"],
        )

    def _init_weights(self, x):
        if type(x) == nn.Linear:
            nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)

    def construct_network(self, host_network_biases_size):
        self._access_config_params()

        self.entry_layer_mlp = nn.Linear(
            self._model_state_representation_size, host_network_biases_size
        )

        self.final_layer_mlp = nn.Sequential(
            nn.Linear(
                self._model_state_representation_size, self._model_hidden_layer_size
            ),
            nn.ReLU(),
            nn.Linear(self._model_hidden_layer_size, host_network_biases_size),
        )

        # Apply Xavier initialisation by recursive search
        self.apply(self._init_weights)

    def forward(self, state_representation: torch.Tensor):
        entry_layer_output = self.entry_layer_mlp(state_representation)
        final_layer_output = self.final_layer_mlp(state_representation)
        return entry_layer_output, final_layer_output
