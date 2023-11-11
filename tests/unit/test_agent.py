import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qmix.agent import DRQNAgent
from qmix.networks import DRQN


@pytest.fixture
def agent_config():
    return {
        "model": {
            "type": "dict",
            "choice": {
                "observation_size": 80,
                "n_actions": 14,
                "embedding_size": 32,
                "hidden_state_size": 32,
                "n_q_values": 14,
            },
        },
        "exploration": {
            "epsilon": {"type": "float", "choice": 1.0},
            "epsilon_min": {"type": "float", "choice": 0.05},
            "epsilon_dec": {"type": "float", "choice": 0.99},
        },
    }


class TestDRQNAgent:
    @pytest.fixture
    def agents_list(self, agent_config):
        agents = []

        target_drqn = DRQN(agent_config)
        target_drqn.construct_network(8)
        online_drqn = copy.deepcopy(target_drqn)

        for i in range(8):  # Creating 8 agents
            agent = DRQNAgent(
                agent_unique_id=i,
                agent_configuration=agent_config,
                num_actions=14,
                target_drqn_network=target_drqn,
                online_drqn_network=online_drqn,
                num_agents=8,
                device="cpu",
            )
            agents.append(agent)

        return agents

    def test_agents_initialization(self, agents_list):
        for agent in agents_list:
            assert isinstance(
                agent, DRQNAgent
            ), f"Initialization failed for DRQNAgent with ID {agent._agent_unique_id}."

    def test_lstm_params_initialization(self, agents_list):
        for agent in agents_list:
            lstm_layer = agent._online_drqn_network.lstm
            assert isinstance(
                lstm_layer, nn.LSTMCell
            ), f"Agent with ID {agent._agent_unique_id} does not have an LSTM layer."

            # Checking if the LSTM weights and biases are initialized
            assert (
                lstm_layer.weight_ih is not None
            ), f"LSTM weight_ih not initialized for agent with ID {agent._agent_unique_id}."
            assert (
                lstm_layer.weight_hh is not None
            ), f"LSTM weight_hh not initialized for agent with ID {agent._agent_unique_id}."
            assert (
                lstm_layer.bias_ih is not None
            ), f"LSTM bias_ih_l0 not initialized for agent with ID {agent._agent_unique_id}."
            assert (
                lstm_layer.bias_hh is not None
            ), f"LSTM bias_hh_l0 not initialized for agent with ID {agent._agent_unique_id}."

    def test_estimate_q_values(self, agents_list):
        for agent in agents_list:
            observation = torch.randn((1, 32, 80))
            prev_action = torch.zeros((1, 32, 14))

            agent_id_one_hot = agent.access_agent_one_hot_id()
            agent_id_one_hot = agent_id_one_hot.unsqueeze(0).unsqueeze(0)
            agent_id_one_hot = agent_id_one_hot.expand(
                observation.shape[0], observation.shape[1], -1
            )

            # Ensure the tensor can be concatenated on the appropriate dimension
            agent_observation = torch.cat([observation, agent_id_one_hot], dim=-1)

            initial_hidden_state = agent._hidden_state.clone()
            initial_cell_state = agent._cell_state.clone()

            q_values = agent.estimate_q_values(agent_observation, prev_action)

            assert q_values.shape == (32, 14), "Unexpected Q-values shape."
            assert agent._hidden_state.shape == (
                agent._online_drqn_network.hidden_state_dim,
            ), "Unexpected hidden state shape."
            assert agent._cell_state.shape == (
                agent._online_drqn_network.hidden_state_dim,
            ), "Unexpected cell state shape."
            assert not torch.equal(
                agent._hidden_state, initial_hidden_state
            ), "Hidden state was not updated."
            assert not torch.equal(
                agent._cell_state, initial_cell_state
            ), "Cell state was not updated."

    def test_estimate_target_q_values(self, agents_list):
        for agent in agents_list:
            observation = torch.randn((1, 32, 80))
            prev_action = torch.zeros((1, 32, 14))

            agent_id_one_hot = agent.access_agent_one_hot_id()
            agent_id_one_hot = agent_id_one_hot.unsqueeze(0).unsqueeze(0)
            agent_id_one_hot = agent_id_one_hot.expand(
                observation.shape[0], observation.shape[1], -1
            )

            # Ensure the tensor can be concatenated on the appropriate dimension
            agent_observation = torch.cat([observation, agent_id_one_hot], dim=-1)

            initial_hidden_state = agent._target_hidden_state.clone()
            initial_cell_state = agent._target_cell_state.clone()

            q_values = agent.estimate_target_q_values(agent_observation, prev_action)

            assert q_values.shape == (32, 14), "Unexpected Q-values shape."
            assert agent._target_hidden_state.shape == (
                agent._online_drqn_network.hidden_state_dim,
            ), "Unexpected target hidden state shape."
            assert agent._target_cell_state.shape == (
                agent._online_drqn_network.hidden_state_dim,
            ), "Unexpected target cell state shape."
            assert not torch.equal(
                agent._target_hidden_state, initial_hidden_state
            ), "Target hidden state was not updated."
            assert not torch.equal(
                agent._target_cell_state, initial_cell_state
            ), "Target cell state was not updated."

    def test_act(self, agents_list):
        for agent in agents_list:
            observation = torch.randn((1, 80))
            available_actions = [1 for x in range(14)]

            agent_id_one_hot = agent.access_agent_one_hot_id()
            agent_id_one_hot = agent_id_one_hot.unsqueeze(0)

            # Ensure the tensor can be concatenated on the appropriate dimension
            agent_observation = torch.cat([observation, agent_id_one_hot], dim=-1)

            # When evaluate is False, action should be one of the available actions
            action = agent.act(agent_observation, available_actions, evaluate=False)
            assert action in range(
                len(available_actions)
            ), f"Action {action} is not within the expected range of available actions for agent with ID {agent._agent_unique_id}"

            # When evaluate is True, the action should also be one of the available actions
            # This assumes that evaluate=True forces deterministic behavior, which may not be the case depending on implementation
            action = agent.act(agent_observation, available_actions, evaluate=True)
            assert action in range(
                len(available_actions)
            ), f"Action {action} is not within the expected range of available actions for agent with ID {agent._agent_unique_id} when evaluating"
