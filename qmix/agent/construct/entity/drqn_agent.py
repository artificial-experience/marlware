import numpy as np
import torch
import torch.functional as F
import torch.optim as optim

from qmix.agent.networks import DRQN
from qmix.common import constants
from qmix.common import methods


class DRQNAgent:
    def __init__(
        self,
        agent_configuration: dict,
        num_actions: int,
        target_drqn_network: DRQN,
        online_drqn_network: DRQN,
    ):
        self._agent_configuration = agent_configuration

        self._online_drqn_network = online_drqn_network
        self._target_drqn_network = target_drqn_network

        # Action space required for act method logic
        self._num_actions = num_actions
        self._action_space = [action for action in range(self._num_actions)]

        # Intrinsic const parameters
        self._training_lr = None
        self._training_gamma = None
        self._training_epsilon_min = None
        self._training_num_epsilon_dec_steps = None
        self._target_network_update_schedule = None

        # Intrinsic non-const parameters
        self._training_epsilon = None
        self._optimizer_step = 0

    def _access_config_params(self):
        self._training_lr = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["training", "lr", "choice"],
        )
        self._training_gamma = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["training", "gamma", "choice"],
        )
        self._training_epsilon_max = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["training", "epsilon_max", "choice"],
        )
        self._training_epsilon_min = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["training", "epsilon_min", "choice"],
        )
        self._training_num_epsilon_dec_steps = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["training", "num_epsilon_dec_steps", "choice"],
        )
        self._training_target_network_update_schedule = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["training", "target_network_update_schedule", "choice"],
        )

    def _decrement_epsilon(self):
        pass

    def _update_target_network_params(self):
        pass

    def fit(self, data: np.ndarray):
        pass

    def act(self, joint_observation: np.ndarray):
        if np.random.random() > self._training_epsilon:
            observation, prev_action = joint_observation
            torch_observation = torch.tensor([observation], dtype=torch.float).to(
                self._target_drqn_network.accelerator_device
            )
            torch_prev_action = torch.tensor([prev_action], dtype=torch.float).to(
                self._target_drqn_network.accelerator_device
            )
            initial_hidden_state = torch.tensor([0.0], dtype=torch.float).to(
                self._target_drqn_network.accelerator_device
            )
            initial_cell_state = torch.tensor([0.0], dtype=torch.float).to(
                self._target_drqn_network.accelerator_device
            )
            q_values, (updated_hidden, updated_cell) = self._online_drqn_network(
                torch_observation,
                torch_prev_action,
                initial_hidden_state,
                initial_cell_state,
            )
            action = torch.argmax(q_values).item()
        else:
            action = np.random.choice(self._action_space)

        return action
