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

        self._model_n_q_values = None
        self._model_hidden_layer_size = None

        self._biases_hypernetwork = None
        self._weights_hypernetwork = None

    def _access_config_params(self):
        # Model configuration
        self._model_n_q_values = methods.get_nested_dict_field(
            directive=self._mixing_network_configuration,
            keys=["model", "choice", "n_q_values"],
        )
        self._model_hidden_layer_size = methods.get_nested_dict_field(
            directive=self._mixing_network_configuration,
            keys=["model", "choice", "hidden_layer_size"],
        )

    def _construct_hypernetworks(
        self,
        hypernetwork_configuration,
        host_network_weights_hidden_size,
        host_network_weights_output_size,
        host_network_biases_hidden_size,
        host_network_biases_output_size,
    ):
        bias_hypernet_config = hypernetwork_configuration.get("bias_updates", None)
        weigh_hypernet_config = hypernetwork_configuration.get("weight_updates", None)

        self._biases_hypernetwork = HyperNetworkForBiases(
            hypernetwork_configuration=bias_hypernet_config
        ).construct_network(
            host_network_biases_hidden_size=host_network_biases_hidden_size,
            host_network_biases_output_size=host_network_biases_output_size,
        )
        self._weights_hypernetwork = HyperNetworkForWeights(
            hypernetwork_configuration=weigh_hypernet_config
        ).construct_network(
            host_network_weights_hidden_size=host_network_weights_hidden_size,
            host_network_weights_output_size=host_network_weights_output_size,
        )

    def _init_weights(self, x):
        if type(x) == nn.Linear:
            nn.init.xavier_uniform_(x.weight)
            x.bias.data.fill_(0.01)

    def _update_network_params(
        self,
        hidden_layer_weights,
        output_layer_weights,
        hidden_layer_biases,
        output_layer_biases,
    ):
        # Reshape operation
        hidden_weights = hidden_layer_weights.view_as(self.hidden_layer.weight)
        output_weights = output_layer_weights.view_as(self.output_layer.weight)
        hidden_biases = hidden_layer_biases.view_as(self.hiden_layer.bias)
        output_biases = output_layer_biases.view_as(self.output_layer.bias)

        with torch.no_grad():
            self.hidden_layer.weight.copy_(hidden_weights)
            self.output_layer.weight.copy_(output_weights)
            self.hidden_layer.bias.copy_(hidden_biases)
            self.output_layer.bias.copy_(output_biases)

    def construct_network(self):
        self._access_config_params()

        self.hidden_layer = nn.Linear(
            self._model_n_q_values, self._model_hidden_layer_size
        )
        self.output_layer = nn.Linear(self._model_hidden_layer_size, 1)  # Output Q_tot

        mixing_network_hidden_layer_weights = self.hidden_layer.weight.numel()
        mixing_network_output_layer_weights = self.output_layer.weight.numel()
        mixing_network_hidden_layer_biases = self.hidden_layer.bias.numel()
        mixing_network_output_layer_biases = self.output_layer.bias.numel()

        self._construct_hypernetworks(
            hypernetwork_configuration=self._hypernetwork_configuration,
            host_network_weights_hidden_size=mixing_network_hidden_layer_weights,
            host_network_weights_output_size=mixing_network_output_layer_weights,
            host_network_biases_hidden_size=mixing_network_hidden_layer_biases,
            host_network_biases_output_size=mixing_network_output_layer_biases,
        )

        # Apply Xavier initialisation by recursive search
        self.apply(self._init_weights)

        return self

    def forward(self, q_values: torch.Tensor, state_representation: torch.Tensor):
        # hypernetwork feed forward
        (
            hypernetwork_hidden_weights,
            hypernetwork_output_weights,
        ) = self._weights_hypernetwork(state_representation)
        (
            hypernetwork_hidden_biases,
            hypernetwork_output_biases,
        ) = self._biases_hypernetwork(state_representation)

        self._update_network_params(
            hidden_layer_weights=hypernetwork_hidden_weights,
            output_layer_weights=hypernetwork_output_weights,
            hidden_layer_biases=hypernetwork_hidden_biases,
            output_layer_biases=hypernetwork_output_biases,
        )

        output = self.hidden_layer(q_values)
        output = nn.ELU(output)

        q_tot = self.output_layer(output)
        return q_tot
