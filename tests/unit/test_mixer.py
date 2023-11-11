import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qmix.networks import MixingNetwork  # Update with your actual import


@pytest.fixture
def mixing_network_config():
    return {
        "mixing": {"model": {"choice": {"n_q_values": 4, "hidden_layer_size": 6}}},
        "hyper_net": {
            "model": {
                "type": "dict",
                "choice": {"state_representation_size": 10, "hidden_layer_size": 6},
            }
        },
    }


class TestMixingNetwork:
    @pytest.fixture
    def mixing_network_instance(self, mixing_network_config):
        mixing_network = MixingNetwork(
            mixing_network_config["mixing"], mixing_network_config["hyper_net"]
        )
        return mixing_network

    def test_initialization(self, mixing_network_instance):
        # Test if the instance is correctly initialized
        assert isinstance(
            mixing_network_instance, nn.Module
        ), "MixingNetwork should be an instance of nn.Module"
        assert hasattr(
            mixing_network_instance, "_mixing_head_configuration"
        ), "MixingNetwork should have a _mixing_head_configuration attribute"
        assert hasattr(
            mixing_network_instance, "_hypernetwork_configuration"
        ), "MixingNetwork should have a _hypernetwork_configuration attribute"

    def test_construct_network(self, mixing_network_instance):
        num_agents = 8
        mixing_network_instance.construct_network(num_agents)

        # Check if the layers are correctly constructed
        assert hasattr(
            mixing_network_instance, "hyper_w_1"
        ), "Network should have hyper_w_1 layer"
        assert hasattr(
            mixing_network_instance, "hyper_b_1"
        ), "Network should have hyper_b_1 layer"
        assert hasattr(
            mixing_network_instance, "hyper_w_final"
        ), "Network should have hyper_w_final layer"
        assert hasattr(mixing_network_instance, "V"), "Network should have V layer"

        # Check parameters shapes
        state_dim = mixing_network_instance.state_dim
        embed_dim = mixing_network_instance.embed_dim

        assert mixing_network_instance.hyper_w_1.weight.size() == (
            embed_dim * num_agents,
            state_dim,
        ), "hyper_w_1 weight matrix should have shape (embed_dim * num_agents, state_dim)"
        assert mixing_network_instance.hyper_b_1.weight.size() == (
            embed_dim,
            state_dim,
        ), "hyper_b_1 weight matrix should have shape (embed_dim, state_dim)"
        assert mixing_network_instance.hyper_w_final.weight.size() == (
            embed_dim,
            state_dim,
        ), "hyper_w_final weight matrix should have shape (embed_dim, state_dim)"
        assert mixing_network_instance.V[0].weight.size() == (
            embed_dim,
            state_dim,
        ), "V layer's first linear module should have weight matrix of shape (embed_dim, state_dim)"
        assert mixing_network_instance.V[2].weight.size() == (
            1,
            embed_dim,
        ), "V layer's second linear module should have weight matrix of shape (1, embed_dim)"

    def test_forward(self, mixing_network_instance):
        batch_size = 2
        num_agents = 8

        mixing_network_instance.construct_network(num_agents)

        state_dim = mixing_network_instance.state_dim

        # Mock data to represent agent_qs and states
        agent_qs = torch.randn(batch_size, num_agents, 1)
        states = torch.randn(1, batch_size, state_dim)

        # Run forward pass
        q_total = mixing_network_instance.forward(agent_qs, states)

        # Verify output shape
        expected_shape = (batch_size, 1, 1)
        assert (
            q_total.size() == expected_shape
        ), f"Output shape should be {expected_shape}, but got {q_total.size()}"

        # Verify output type
        assert isinstance(
            q_total, torch.Tensor
        ), "Output of forward should be a torch.Tensor"


class TestMixingNetworkCalculation:
    @pytest.fixture
    def mixing_network_instance(self, mixing_network_config):
        mixing_network = MixingNetwork(
            mixing_network_config["mixing"], mixing_network_config["hyper_net"]
        )
        mixing_network.construct_network(8)  # Construct network for 8 agents
        return mixing_network

    def test_known_calculation(self, mixing_network_instance):
        # Set all weights and biases in the network to 1 and 0 respectively.
        batch_size = 2

        for param in mixing_network_instance.parameters():
            nn.init.constant_(param, val=1.0 if param.dim() > 1 else 0.0)

        # Define a controlled input
        agent_qs = torch.ones(
            batch_size, 8, 1
        )  # Batch size 32, 8 agents, single Q value for each agent.
        states = torch.ones(1, batch_size, 10)  # Batch size 32, state dimension 168.

        # Expected output calculation
        # Since we set all weights to 1 and biases to 0, the output for each batch item should be
        # the sum of agent_qs plus the V(s) output, which will also be a constant because V(s) has weights of 1 and bias of 0.
        expected_output = torch.full(
            (batch_size, 1, 1), 8 + 1, dtype=torch.float32
        )  # Each agent contributes 1, and V(s) contributes 1

        # Run the forward pass
        actual_output = mixing_network_instance.forward(agent_qs, states)

        # Check if the actual output matches the expected output
        assert torch.allclose(
            actual_output, expected_output
        ), "The output of the network did not match the expected output."
