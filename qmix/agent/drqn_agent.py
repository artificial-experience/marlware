import numpy as np
import torch

from qmix.common import methods
from qmix.networks import DRQN
from qmix.selector import EpsilonGreedyActionSelector


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

    """

    def __init__(
        self,
        agent_unique_id: int,
        agent_configuration: dict,
        num_actions: int,
        target_drqn_network: DRQN,
        online_drqn_network: DRQN,
        num_agents: int,
        device: str,
    ):
        self._num_agents = num_agents
        self._agent_configuration = agent_configuration

        # Online and target nets
        self._online_drqn_network = online_drqn_network
        self._target_drqn_network = target_drqn_network

        # Action space required for act method logic
        self._num_actions = num_actions
        self._action_space = [action for action in range(self._num_actions)]

        self._acceleration_device = device

        # Agent unique identifier
        self._agent_unique_id = torch.tensor(
            agent_unique_id, device=self._acceleration_device
        )

        # Epsilon params
        self._epsilon_start = None
        self._epsilon_min = None
        self._epsilon_anneal_time = None

        # Action selector
        self._action_selector = None

        # Intrinsic non-const parameters
        self._hidden_state = torch.zeros(
            self._online_drqn_network.hidden_state_dim, device=self._acceleration_device
        )
        self._cell_state = torch.zeros(
            self._online_drqn_network.hidden_state_dim, device=self._acceleration_device
        )
        self._prev_action = torch.zeros(
            1, len(self._action_space), device=self._acceleration_device
        )

        self._target_hidden_state = torch.zeros(
            self._online_drqn_network.hidden_state_dim, device=self._acceleration_device
        )
        self._target_cell_state = torch.zeros(
            self._online_drqn_network.hidden_state_dim, device=self._acceleration_device
        )

    def _access_config_params(self):
        """acces configuration parameters"""
        self._epsilon_start = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["exploration", "epsilon_start", "choice"],
        )
        self._epsilon_min = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["exploration", "epsilon_min", "choice"],
        )
        self._epsilon_anneal_time = methods.get_nested_dict_field(
            directive=self._agent_configuration,
            keys=["exploration", "epsilon_anneal_time", "choice"],
        )

    def set_action_selector(self):
        self._access_config_params()
        self._action_selector = EpsilonGreedyActionSelector(
            epsilon_start=self._epsilon_start,
            epsilon_finish=self._epsilon_min,
            epsilon_anneal_time=self._epsilon_anneal_time,
        )

    def reset_intrinsic_lstm_params(self):
        """Reset the agent's memory (LSTM states and previous action)"""
        self._hidden_state = torch.zeros(
            self._online_drqn_network.hidden_state_dim, device=self._acceleration_device
        )
        self._cell_state = torch.zeros(
            self._online_drqn_network.hidden_state_dim, device=self._acceleration_device
        )

        self._target_hidden_state = torch.zeros(
            self._target_drqn_network.hidden_state_dim, device=self._acceleration_device
        )
        self._target_cell_state = torch.zeros(
            self._target_drqn_network.hidden_state_dim, device=self._acceleration_device
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

    def access_agent_one_hot_id(self):
        """Access one hot id for the agent - that is added to the observation"""
        self._agent_unique_id = self._agent_unique_id.to(self._acceleration_device)
        agent_one_hot_id = torch.nn.functional.one_hot(
            self._agent_unique_id,
            num_classes=self._num_agents,
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
        self, target_observation: torch.Tensor, target_prev_action: torch.Tensor
    ):
        """Estiamte target action value functions"""

        target_q_values, target_lstm_memory = self._target_drqn_network(
            observation=target_observation,
            prev_action=target_prev_action,
            hidden_state=self._target_hidden_state,
            cell_state=self._target_cell_state,
        )
        self._target_hidden_state, self._target_cell_state = target_lstm_memory
        return target_q_values

    def reset_prev_action(self):
        """reset information about previously taken action"""
        self._prev_action = torch.zeros(
            1, len(self._action_space), device=self._acceleration_device
        )

    def act(
        self,
        observation: np.ndarray,
        available_actions: list,
        t_env: int,
        evaluate: bool,
    ):
        """
        Produce an action for the given observation using the EpsilonGreedyActionSelector
        """
        # Convert observation to tensor and add batch dimension (batch size = 1)
        observation_tensor = (
            torch.tensor(observation).float().to(self._acceleration_device).unsqueeze(0)
        )

        # Convert available actions to tensor
        avail_actions_tensor = (
            torch.tensor(available_actions)
            .float()
            .to(self._acceleration_device)
            .unsqueeze(0)
        )

        # Compute Q-values using the network (assuming self._online_drqn_network is your Q-network)
        q_values, lstm_memory = self._online_drqn_network(
            observation_tensor, self._prev_action, self._hidden_state, self._cell_state
        )

        # Use the selector to choose the action
        action = self._action_selector.select_action(
            q_values, avail_actions_tensor, t_env, test_mode=evaluate
        ).item()

        # Update hidden states if your model uses LSTM or any other stateful layers
        self._hidden_state, self._cell_state = lstm_memory

        # Store the current action as the previous action for the next step, one-hot encoded
        self._prev_action = torch.nn.functional.one_hot(
            torch.tensor([action], device=self._acceleration_device),
            num_classes=len(self._action_space),
        )

        return action
