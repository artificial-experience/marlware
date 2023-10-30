import numpy as np
import torch

from qmix.common import methods
from qmix.networks import DRQN


class DRQNAgent:
    """
    Deep recurrent q-network agent
    Single agent instance that has a shared drqn network

    Args:
        param: [agent_unique_id]: agents unique identifier
        param: [agent_configuration]: agent hyperparameter configuration
        param: [num_actions]: number of actions available in the env
        param: [target_drqn_network]: network object to be used in order to calculate target
        param: [online_drqn_network]: network object to be used in order to produce actions
        param: [num_agents]: number of agents in the interaction zone

    Internal State:
        param: [action_space]: list of all actions available in the env
        param: [epsilon]: exploration hyperparameter
        param: [epsilon_min]: minimum exploration value
        param: [epsilon_dec]: decay param for each epsilon update
        param: [hidden_state]: keeps track of hidden state weights for online net
        param: [cell_state]: heeps track of cell state weights for online net
        param: [prev_action]: heeps track of previous action
        param: [hidden_state]: keeps track of hidden state weights for target net
        param: [cell_state]: heeps track of cell state weights for target net

    Internal State: []
        param: []
    """

    def __init__(
        self,
        agent_unique_id: int,
        agent_configuration: dict,
        num_actions: int,
        target_drqn_network: DRQN,
        online_drqn_network: DRQN,
        num_agents: int,
    ):
        self._agent_unique_id = torch.tensor(agent_unique_id)
        self._num_agents = num_agents
        self._agent_configuration = agent_configuration

        # Online and target nets
        self._online_drqn_network = online_drqn_network
        self._target_drqn_network = target_drqn_network

        # Action space required for act method logic
        self._num_actions = num_actions
        self._action_space = [action for action in range(self._num_actions)]

        # Epsilon params
        self._epsilon = 1.0
        self._epsilon_min = 0.05
        self._epsilon_dec = 0.99

        # Intrinsic non-const parameters
        self._hidden_state = torch.zeros(self._online_drqn_network.hidden_state_dim)
        self._cell_state = torch.zeros(self._online_drqn_network.hidden_state_dim)
        self._prev_action = torch.zeros(1, len(self._action_space))

        self._target_hidden_state = torch.zeros(
            self._online_drqn_network.hidden_state_dim
        )
        self._target_cell_state = torch.zeros(
            self._online_drqn_network.hidden_state_dim
        )

    def _access_config_params(self):
        """acces configuration parameters"""
        self._epsilon = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["exploration", "epsilon", "choice"],
        )
        self._epsilon_min = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["exploration", "epsilon_min", "choice"],
        )
        self._epsilon_dec = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["exploration", "epsilon_dec", "choice"],
        )

    def reset_intrinsic_lstm_params(self):
        """Reset the agent's memory (LSTM states and previous action)"""
        self._hidden_state = torch.zeros(self._online_drqn_network.hidden_state_dim)
        self._cell_state = torch.zeros(self._online_drqn_network.hidden_state_dim)

        self._target_hidden_state = torch.zeros(
            self._target_drqn_network.hidden_state_dim
        )
        self._target_cell_state = torch.zeros(
            self._target_drqn_network.hidden_state_dim
        )

    def update_target_network_params(self, tau=1.0):
        """
        Copies the weights from the online network to the target network.
        If tau is 1.0 (default), it's a hard update. Otherwise, it's a soft update.

        :param tau: The soft update factor, if < 1.0. Default is 1.0 (hard update).
        """
        for target_param, online_param in zip(
            self._target_drqn_network.parameters(),
            self._online_drqn_network.parameters(),
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def decrease_exploration(self):
        """Decrease exploration parameters"""
        decreased_epsilon = self._epsilon * self._epsilon_dec
        self._epsilon = max(self._epsilon_min, decreased_epsilon)

    def access_agent_one_hot_id(self):
        """Access one hot id for the agent - that is added to the observation"""
        agent_one_hot_id = torch.nn.functional.one_hot(
            self._agent_unique_id, num_classes=self._num_agents
        )
        return agent_one_hot_id

    def estimate_q_values(self, observation: torch.Tensor, prev_action: torch.Tensor):
        """Estiamte Q onlie action value functions"""
        q_values, lstm_memory = self._online_drqn_network(
            observation=observation,
            prev_action=prev_action,
            hidden_state=self._hidden_state,
            cell_state=self._cell_state,
        )
        self._hidden_state, self._cell_state = lstm_memory
        return q_values

    def estimate_target_q_values(
        self, target_observation: torch.Tensor, prev_action: torch.Tensor
    ):
        """Estiamte target action value functions"""
        with torch.no_grad():
            target_q_values, target_lstm_memory = self._target_drqn_network(
                observation=target_observation,
                prev_action=prev_action,
                hidden_state=self._target_hidden_state,
                cell_state=self._target_cell_state,
            )
            self._target_hidden_state, self._target_cell_state = target_lstm_memory
        return target_q_values

    def act(
        self, observation: np.ndarray, available_actions: np.ndarray, evaluate: bool
    ):
        """Produce epsilon-greedy action given observation"""
        if evaluate or np.random.random() > self._epsilon:
            observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)

            q_values, lstm_memory = self._online_drqn_network(
                observation,
                self._prev_action,
                self._hidden_state,
                self._cell_state,
            )

            # Mask unavailable actions
            masked_q_values = q_values.clone()
            masked_q_values[available_actions == 0.0] = -float("inf")

            # Select action with the highest Q-value among available actions
            action = torch.argmax(masked_q_values).item()

            self._hidden_state, self._cell_state = lstm_memory
            self._prev_action = torch.nn.functional.one_hot(
                torch.tensor([action]), num_classes=len(self._action_space)
            )

        else:
            # Random action selection among available actions
            available_actions = np.nonzero(available_actions)[0]
            action = np.random.choice(available_actions)

        return action
