import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qmix.networks import DRQN


@pytest.fixture
def drqn_config():
    return {
        "model": {
            "type": "dict",
            "choice": {
                "observation_size": 80,
                "n_actions": 14,
                "embedding_size": 64,
                "hidden_state_size": 128,
                "n_q_values": 14,
            },
        }
    }


class TestDRQNNetworkForEightAgents:
    @pytest.fixture
    def drqn_instance(self, drqn_config):
        drqn = DRQN(drqn_config)
        drqn.construct_network(8)  # Assuming 8 agents now
        return drqn

    def test_forward_propagation(self, drqn_instance):
        observation = torch.rand((2, 80))  # Corrected observation size to 80

        # Agent identifiers for batch size of 2 for the first two agents
        agent_ids = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])
        # Actions from a batch of size 2 for 14 possible actions
        prev_action = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        input_tensor = torch.cat([observation, agent_ids], dim=1)

        hidden_state = torch.rand(drqn_instance.hidden_state_dim)
        cell_state = torch.rand(drqn_instance.hidden_state_dim)

        q_values, (updated_hidden_state, updated_cell_state) = drqn_instance(
            input_tensor, prev_action, hidden_state, cell_state
        )

        assert q_values.size() == (2, drqn_instance._model_n_q_values)
        assert updated_hidden_state.size() == (drqn_instance.hidden_state_dim,)
        assert updated_cell_state.size() == (drqn_instance.hidden_state_dim,)

    def test_hidden_state_update(self, drqn_instance):
        observation = torch.rand((2, 80))  # Observation for batch size of 2

        # Actions for a batch of size 2 for 14 possible actions
        prev_action = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        # Agent identifiers for batch size of 2 for the first two agents
        agent_ids = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])

        # Combine observation, action, and agent ID into a single input tensor
        input_tensor = torch.cat([observation, agent_ids], dim=1)

        hidden_state = torch.rand(drqn_instance.hidden_state_dim)
        cell_state = torch.rand(drqn_instance.hidden_state_dim)

        _, (updated_hidden_state1, updated_cell_state1) = drqn_instance(
            input_tensor, prev_action, hidden_state, cell_state
        )
        _, (updated_hidden_state2, updated_cell_state2) = drqn_instance(
            input_tensor, prev_action, hidden_state, cell_state
        )

        # Ensure the hidden and cell states are updated correctly
        assert torch.equal(updated_hidden_state1, updated_hidden_state2)
        assert torch.equal(updated_cell_state1, updated_cell_state2)

    def test_weight_updates(self, drqn_instance):
        # 1. Get the initial weights of the DRQN instance
        initial_weights = {
            name: param.clone() for name, param in drqn_instance.named_parameters()
        }

        # Dummy data for forward pass
        observation = torch.rand((2, 80))
        agent_ids = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])
        prev_action = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        input_tensor = torch.cat([observation, agent_ids], dim=1)

        hidden_state = torch.rand(drqn_instance.hidden_state_dim)
        cell_state = torch.rand(drqn_instance.hidden_state_dim)

        # Forward pass
        optimizer = torch.optim.SGD(
            drqn_instance.parameters(), lr=0.01
        )  # Define an optimizer
        optimizer.zero_grad()  # Reset any prior gradients
        q_values, _ = drqn_instance(input_tensor, prev_action, hidden_state, cell_state)

        # 2. Compute a dummy loss - use MSELoss in this case
        target = torch.rand_like(q_values)
        loss = F.mse_loss(q_values, target)

        # 3. Backpropagate the loss
        loss.backward()
        optimizer.step()

        # 4. Check the weights again
        for name, param in drqn_instance.named_parameters():
            assert not torch.equal(
                initial_weights[name], param
            )  # Assert that weights have indeed changed
