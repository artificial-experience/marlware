import torch
import torch.nn as nn

from qmix.common import methods
from qmix.networks import HyperNetworkForBiases
from qmix.networks import HyperNetworkForWeights


class MixingNetwork(nn.Module):
    def __init__(
        self, mixing_network_configuration: dict, hypernetwork_configuration: dict
    ):
        """
        Mixer network that thakes q-values for each agent and compresses them
        into a single cummulative q-tot value

        Args:
            :param [mixing_network_configuration]: configuration dictionary for mixier
            :param [hypernetwork_configuration]: configuration dictionary for weights and biases hypernets

        Internal State:
            :param [model_n_q_values]: shape of n_q_values tensor
            :param [model_hidden_layer_size]: hidden layer hyperparameter size for mixer
            :param [biases_hypernetowrk]: entity of biases hypernetowrk class
            :param [weights_hypernetwork]: entity of weights hypernetowrk class
        """
        super().__init__()
        self._mixing_network_configuration = mixing_network_configuration
        self._hypernetwork_configuration = hypernetwork_configuration

        self._model_n_q_values = None
        self._model_hidden_layer_size = None

        self._biases_hypernetwork = None
        self._weights_hypernetwork = None

    def _access_config_params(self):
        """extract values given config dict"""
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
        """create hypernetwork for biases and weights"""
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
        """weight initializer method - xavier"""
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
        """insert weights and biases from hypernetwork into mixer"""
        hidden_weights = hidden_layer_weights.view_as(self.hidden_layer.weight)
        output_weights = output_layer_weights.view_as(self.output_layer.weight)
        hidden_biases = hidden_layer_biases.view_as(self.hidden_layer.bias)
        output_biases = output_layer_biases.view_as(self.output_layer.bias)

        with torch.no_grad():
            self.hidden_layer.weight.copy_(hidden_weights)
            self.output_layer.weight.copy_(output_weights)
            self.hidden_layer.bias.copy_(hidden_biases)
            self.output_layer.bias.copy_(output_biases)

    def construct_network(self, num_agents: int):
        """construct hypernetworks and mixer networks"""
        self._access_config_params()

        self.hidden_layer = nn.Linear(num_agents, self._model_hidden_layer_size)
        self.elu_activation = nn.ELU()
        self.output_layer = nn.Linear(self._model_hidden_layer_size, 1)

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
        """given q-values from all agents calculate the q-tot"""
        (
            hypernetwork_hidden_weights,
            hypernetwork_output_weights,
        ) = self._weights_hypernetwork(state_representation)
        (
            hypernetwork_hidden_biases,
            hypernetwork_output_biases,
        ) = self._biases_hypernetwork(state_representation)

        # Prepare hypernetwork outputs for mixing network parameters
        hidden_weights_squeezed = hypernetwork_hidden_weights.squeeze(0)
        output_weights_squeezed = hypernetwork_output_weights.squeeze(0)

        hidden_biases_squeezed = hypernetwork_hidden_biases.squeeze(0)
        output_biases_squeezed = hypernetwork_output_biases.squeeze(0)

        collective_q_tot = []
        for batch_id in range(q_values.size(0)):
            # Use clone to create new layers with learned weights and biases
            hidden_layer = nn.Linear(
                self.hidden_layer.in_features, self.hidden_layer.out_features
            )
            hidden_layer.weight.data = hidden_weights_squeezed[batch_id].view_as(
                hidden_layer.weight
            )
            hidden_layer.bias.data = hidden_biases_squeezed[batch_id].view_as(
                hidden_layer.bias
            )

            output_layer = nn.Linear(
                self.output_layer.in_features, self.output_layer.out_features
            )
            output_layer.weight.data = output_weights_squeezed[batch_id].view_as(
                output_layer.weight
            )
            output_layer.bias.data = output_biases_squeezed[batch_id].view_as(
                output_layer.bias
            )

            # Continue with forward pass
            sample_q_values = q_values[batch_id, :, :]
            prepared_sample_q_values = sample_q_values.reshape(-1)
            output = hidden_layer(prepared_sample_q_values)
            output = self.elu_activation(output)

            q_tot = output_layer(output)
            collective_q_tot.append(q_tot)

        t_collective_q_tot = torch.stack(collective_q_tot, dim=0)

        return t_collective_q_tot
