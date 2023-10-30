import torch
import torch.nn as nn

from qmix.common import methods


class MixingNetwork(nn.Module):
    def __init__(self, mixing_network_configuration, hypernetwork_configuration):
        super().__init__()
        self._mixing_network_configuration = mixing_network_configuration
        self._hypernetwork_configuration = hypernetwork_configuration

    def _access_config_params(self):
        """extract values given config dict"""
        self.state_dim = methods.get_nested_dict_field(
            directive=self._hypernetwork_configuration,
            keys=["weight_updates", "model", "choice", "state_representation_size"],
        )
        self.embed_dim = methods.get_nested_dict_field(
            directive=self._mixing_network_configuration,
            keys=["model", "choice", "hidden_layer_size"],
        )

    def construct_network(self, num_agents: int):
        """given number of agents the method will construct each agent and return itself"""
        self._access_config_params()

        # State-dependent weights for the first layer
        self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * num_agents)
        # State-dependent bias for the first layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # State-dependent weights for the second layer
        self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) as a bias for the final output
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

        return self

    def forward(self, agent_qs, states):
        # Reshape states - get rid of the first dimension as it is  1
        states = states.squeeze(0)

        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)

        w1 = w1.view(-1, 8, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        agent_qs = agent_qs.reshape(-1, 1, 8)

        hidden = torch.nn.functional.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)

        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)

        # Compute final output
        y = torch.bmm(hidden, w_final) + v

        # Reshape and return
        q_tot = y.view(agent_qs.size(0), -1, 1)
        return q_tot
