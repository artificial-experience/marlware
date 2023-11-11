import copy

import numpy as np
import torch
from tqdm import tqdm

from qmix.abstract.construct import BaseConstruct
from qmix.agent import DRQNAgent
from qmix.common import constants
from qmix.common import methods
from qmix.environment import SC2Environment
from qmix.memory import initialize_memory
from qmix.networks import DRQN
from qmix.networks import MixingNetwork


class QMIXSharedParamsConstruct(BaseConstruct):
    """
    Construct class that instantiates DRQN agents and networks

    Args:
        param: [construct_registry_directive]: directive consstructed by registry

    Internal State:
        param: [construct_configuration]: directive consstructed by registry
        param: [n_agents]: number of agents to be instantiated
        param: [environment]: env to be instantiated
        param: [environment_info]: intrinsic informations about the env at hand
        param: [replay_memory]: replay buffer instance
        param: [shared_target_drqn_network]: shared amongst the agents network that is used for target calculation (frozen)
        param: [shared_online_drqn_network]: shared amongst the agents network that is used for action selection and error calculation
        param: [target_mixing_network]: mixing network that is used for target calculation (frozen)
        param: [online_mixing_network]: mixing network that is used for calculation of error
        param: [learning_rate]: hyperparameter for optimizer
        param: [discount_factor]: hyperparameter for reward discounting
        param: [target_network_update_schedule]: how often to update weights of target nets
        param: [accelerator_device]: either CPU or GPU
        param: [params]: local parameters for optimizer
        param: [optimizer]: local optimizer such as Adam or RMSprop
        param: [criterion]: objective function
    """

    def __init__(self, construct_registry_directive: dict):
        self._construct_registry_directive = construct_registry_directive
        self._construct_configuration = None

        # Instantiated agents
        self._agents = None

        # Instantiated env
        self._environment = None
        self._environment_info = None

        # Instantiated memory buffer
        self._replay_memory = None

        # Networks
        self._shared_target_drqn_network = None
        self._shared_online_drqn_network = None

        self._target_mixing_network = None
        self._online_mixing_network = None

        # Intrinsic const parameters
        self._learning_rate = None
        self._discount_factor = None
        self._target_network_update_schedule = None
        self._grad_clip = None

        # Default device set to None
        self._accelerator_device = None

        # Parameters
        self._params = None

        # Construct optimizer and criterion
        self._optimizer = None
        self._criterion = None

        self._timesteps = 0

    @classmethod
    def from_construct_registry_directive(cls, construct_registry_directive: str):
        """create construct given directve"""
        instance = cls(construct_registry_directive)

        # Get the path to the construct file
        path_to_construct_file = construct_registry_directive.get(
            "path_to_construct_file",
            None,  # move key to constants as it is derived from registration dict
        )
        construct_file_path = (
            constants.Directories.TRAINABLE_CONFIG_DIR.value / path_to_construct_file
        ).absolute()

        # Load the YAML configuration
        configuration = methods.load_yaml(construct_file_path)
        instance._construct_configuration = configuration

        # Extract the accelerator device and number of agents from the configuration
        accelerator_type = methods.get_nested_dict_field(
            directive=configuration,
            keys=[
                "architecture-directive",
                "device_configuration",
                "accelerator",
                "choice",
            ],
        )
        instance._accelerator_device = torch.device(accelerator_type)
        instance._learning_rate = methods.get_nested_dict_field(
            directive=configuration,
            keys=[
                "architecture-directive",
                "construct_configuration",
                "training",
                "lr",
                "choice",
            ],
        )
        instance._discount_factor = methods.get_nested_dict_field(
            directive=configuration,
            keys=[
                "architecture-directive",
                "construct_configuration",
                "training",
                "gamma",
                "choice",
            ],
        )
        instance._target_network_update_schedule = methods.get_nested_dict_field(
            directive=configuration,
            keys=[
                "architecture-directive",
                "construct_configuration",
                "training",
                "target_network_update_schedule",
                "choice",
            ],
        )
        instance._grad_clip = methods.get_nested_dict_field(
            directive=configuration,
            keys=[
                "architecture-directive",
                "construct_configuration",
                "training",
                "grad_clip",
                "choice",
            ],
        )
        return instance

    def _instantiate_optimizer_and_criterion(self):
        """instantiate objective function and optimizer"""
        self._optimizer = torch.optim.RMSprop(
            params=self._params, lr=self._learning_rate
        )
        self._criterion = torch.nn.MSELoss()

    def _instantiate_factorisation_network(
        self,
        hypernetwork_configuration: dict,
        mixing_network_configuration: dict,
        num_agents: int,
    ):
        """create mixer instance for both online and target settings"""
        self._online_mixing_network = MixingNetwork(
            mixing_network_configuration=mixing_network_configuration,
            hypernetwork_configuration=hypernetwork_configuration,
        ).construct_network(num_agents=num_agents)

        self._target_mixing_network = copy.deepcopy(self._online_mixing_network)

        self._target_mixing_network.to(self._accelerator_device)
        self._online_mixing_network.to(self._accelerator_device)

    def _spawn_agents(
        self, drqn_configuration: dict, num_actions: int, num_agents: int
    ):
        """create N agent instances"""
        # Create target and online networks
        self._shared_target_drqn_network = DRQN(
            config=drqn_configuration
        ).construct_network(num_agents=num_agents)
        self._shared_online_drqn_network = copy.deepcopy(
            self._shared_target_drqn_network
        )

        self._shared_target_drqn_network.to(self._accelerator_device)
        self._shared_online_drqn_network.to(self._accelerator_device)

        self._agents = [
            DRQNAgent(
                agent_unique_id=identifier,
                agent_configuration=drqn_configuration,
                num_actions=num_actions,
                target_drqn_network=self._shared_target_drqn_network,
                online_drqn_network=self._shared_online_drqn_network,
                num_agents=num_agents,
                device=self._accelerator_device,
            )
            for identifier in range(num_agents)
        ]

        for agent in self._agents:
            agent.set_action_selector()

    def _instantiate_env(self, environment_configuration: dict):
        """create environment instance given the configuration dict"""
        environment_creator = SC2Environment(config=environment_configuration)
        (
            self._environment,
            self._environment_info,
        ) = environment_creator.create_env_instance()

    def _instantiate_replay_memory(self, memory_configuration: dict):
        """create generic replay memory"""
        max_size = methods.get_nested_dict_field(
            directive=memory_configuration,
            keys=["max_size", "choice"],
        )
        batch_size = methods.get_nested_dict_field(
            directive=memory_configuration,
            keys=["batch_size", "choice"],
        )
        prioritized = methods.get_nested_dict_field(
            directive=memory_configuration,
            keys=["prioritized", "choice"],
        )
        prev_actions_field = "prev_actions"
        prev_actions_vals = np.zeros([max_size, 8], dtype=np.int64)

        avail_actions_field = "avail_actions"
        avail_actions_vals = np.zeros([max_size, 8, 14], dtype=np.int64)

        states_field = "states"
        states_vals = np.zeros([max_size, 168], dtype=np.float32)

        next_states_field = "next_states"
        next_states_vals = np.zeros([max_size, 168], dtype=np.float32)

        extra_fields = (
            prev_actions_field,
            states_field,
            next_states_field,
            avail_actions_field,
        )
        extra_vals = (
            prev_actions_vals,
            states_vals,
            next_states_vals,
            avail_actions_vals,
        )

        observation_shape = (80,)
        self._memory = initialize_memory(
            obs_shape=observation_shape,
            n_actions=14,
            n_agents=8,
            max_size=max_size,
            batch_size=batch_size,
            prioritized=prioritized,
            extra_fields=extra_fields,
            extra_vals=extra_vals,
        )

    def _check_construct(self):
        """check whether construct exists"""
        assert self._agents is not None, "Agents are not spawned"
        assert (
            self._online_mixing_network is not None
        ), "Online mixing network is not instantiated"
        assert (
            self._target_mixing_network is not None
        ), "Target mixing network is not instantiated"
        assert self._environment is not None, "Environment is not instantiated"

    def update_target_network_params(self, tau=1.0):
        """
        Copies the weights from the online network to the target network.
        If tau is 1.0 (default), it's a hard update. Otherwise, it's a soft update.

        :param tau: The soft update factor, if < 1.0. Default is 1.0 (hard update).
        """
        for target_param, online_param in zip(
            self._target_mixing_network.parameters(),
            self._online_mixing_network.parameters(),
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def commit(self):
        """commit updates and check construct"""
        drqn_configuration = methods.get_nested_dict_field(
            directive=self._construct_configuration,
            keys=["architecture-directive", "drqn_configuration"],
        )
        hypernetwork_configuration = methods.get_nested_dict_field(
            directive=self._construct_configuration,
            keys=["architecture-directive", "hypernetwork_configuration"],
        )
        mixing_network_configuration = methods.get_nested_dict_field(
            directive=self._construct_configuration,
            keys=["architecture-directive", "mixing_head_configuration"],
        )
        memory_configuration = methods.get_nested_dict_field(
            directive=self._construct_configuration,
            keys=["memory-directive"],
        )
        environment_configuration = methods.get_nested_dict_field(
            directive=self._construct_configuration,
            keys=["environment-directive"],
        )
        num_actions = methods.get_nested_dict_field(
            directive=self._construct_configuration,
            keys=[
                "architecture-directive",
                "drqn_configuration",
                "model",
                "choice",
                "n_actions",
            ],
        )
        num_agents = methods.get_nested_dict_field(
            directive=self._construct_configuration,
            keys=["environment-directive", "num_agents", "choice"],
        )

        self._instantiate_factorisation_network(
            hypernetwork_configuration=hypernetwork_configuration,
            mixing_network_configuration=mixing_network_configuration,
            num_agents=num_agents,
        )

        self._spawn_agents(
            drqn_configuration=drqn_configuration,
            num_actions=num_actions,
            num_agents=num_agents,
        )

        self._instantiate_env(environment_configuration=environment_configuration)
        self._instantiate_replay_memory(memory_configuration=memory_configuration)

        self._update_construct_parameters()
        self._instantiate_optimizer_and_criterion()

        self._check_construct()

        return self

    def _update_construct_parameters(self):
        """helper method for updaing construct local parameters"""
        parameters = []

        parameters += list(self._shared_target_drqn_network.parameters())
        parameters += list(self._target_mixing_network.parameters())
        parameters += list(self._shared_online_drqn_network.parameters())
        parameters += list(self._online_mixing_network.parameters())

        self._params = parameters

    def close_env(self):
        """self explanatory method"""
        self._environment.close()

    # ==================================
    # Section 1: Rollout Collecting
    # ==================================

    def _initialize_environment(self):
        """reset environment to start settings"""
        self._environment.reset()
        for agent in self._agents:
            agent.reset_prev_action()

    def _fetch_environment_details(self):
        """get observation and state from environment"""
        observations = self._environment.get_obs()
        states = self._environment.get_state()
        return observations, states

    def _get_agent_actions(self, observations, timesteps, evaluate=False):
        """return agent actions along with available actions"""
        actions = []
        avail_actions = []
        num_agents = len(self._agents)

        for agent_id in range(num_agents):
            available_actions = self._environment.get_avail_agent_actions(agent_id)
            avail_actions.append(available_actions)
            available_actions_index = np.nonzero(available_actions)[0]

            agent = self._agents[agent_id]
            agent_observation = self._prepare_agent_observation(
                observations[agent_id], agent
            )
            agent_action = agent.act(
                agent_observation, available_actions, timesteps, evaluate
            )

            if agent_action not in available_actions_index:
                raise SystemError(
                    "Agent chose the unavailable action in the environment"
                )

            actions.append(agent_action)

        return actions, avail_actions

    def _prepare_agent_observation(self, observation, agent):
        """concatenate agent one hot encoded information with observation"""
        agent_one_hot = agent.access_agent_one_hot_id()
        agent_observation = torch.tensor(
            observation, dtype=torch.float32, device=self._accelerator_device
        )
        agent_observation = torch.cat([agent_observation, agent_one_hot], dim=0)
        return agent_observation

    def _execute_actions(self, actions):
        """step the environment given action from all N agents"""
        print(actions)
        reward, terminated, _ = self._environment.step(actions)
        return reward, terminated

    def _store_transitions(
        self,
        observations,
        actions,
        reward,
        next_observations,
        terminated,
        prev_actions,
        states,
        next_states,
        avail_actions,
    ):
        """store transition for each agent in generic buffer"""
        if prev_actions is not None:
            data = [
                observations,
                actions,
                reward,
                next_observations,
                terminated,
                prev_actions,
                states,
                next_states,
                avail_actions,
            ]
            self._memory.store_transition(data)

    def memory_ready(self):
        """self explanatory method"""
        return self._memory.ready()

    def collect_rollouts(self):
        """method to spin environment and collect rollouts / trajectories in order to learn"""
        self._initialize_environment()
        terminated = False
        prev_actions = None
        next_observations = None
        next_states = None

        while not terminated:
            observations, states = self._fetch_environment_details()
            actions, avail_actions = self._get_agent_actions(
                observations, self._timesteps, evaluate=False
            )

            reward, terminated = self._execute_actions(actions)

            next_observations, next_states = self._fetch_environment_details()

            self._store_transitions(
                observations,
                actions,
                reward,
                next_observations,
                terminated,
                prev_actions,
                states,
                next_states,
                avail_actions,
            )
            prev_actions = actions
            self._timesteps += 1

    # ==================================
    # Section 2: Optimization
    # ==================================

    def _convert_to_tensors(self, sample_data):
        """convert sample trajectory batch from numpy array to tensor"""
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            prev_actions,
            states,
            next_states,
            avail_actions,
        ) = sample_data

        tensors = {
            "observations": torch.tensor(
                [observations], dtype=torch.float32, device=self._accelerator_device
            ),
            "actions": torch.tensor(
                [actions], dtype=torch.int64, device=self._accelerator_device
            ),
            "rewards": torch.tensor(
                [rewards], dtype=torch.float32, device=self._accelerator_device
            ),
            "next_observations": torch.tensor(
                [next_observations],
                dtype=torch.float32,
                device=self._accelerator_device,
            ),
            "dones": torch.tensor(
                [dones], dtype=torch.float32, device=self._accelerator_device
            ),
            "prev_actions": torch.tensor(
                [prev_actions], dtype=torch.int64, device=self._accelerator_device
            ),
            "states": torch.tensor(
                [states], dtype=torch.float32, device=self._accelerator_device
            ),
            "next_states": torch.tensor(
                [next_states], dtype=torch.float32, device=self._accelerator_device
            ),
            "avail_actions": torch.tensor(
                [avail_actions], dtype=torch.float32, device=self._accelerator_device
            ),
        }
        return tensors

    def _get_multi_agent_q_values(self, tensors):
        """for each agent collect approximated q-values"""
        multi_agent_q_vals = []
        multi_agent_target_q_vals = []
        for agent_id, agent in enumerate(self._agents):
            agent_q_vals, target_q_vals = self._get_agent_q_values(
                agent_id, agent, tensors
            )
            multi_agent_q_vals.append(agent_q_vals)
            multi_agent_target_q_vals.append(target_q_vals)
        return multi_agent_q_vals, multi_agent_target_q_vals

    def _get_agent_q_values(self, agent_id, agent, tensors):
        """helper method for calculating q-values per each agent"""
        # Implement the logic to get the agent's Q values and target Q values
        agent.reset_intrinsic_lstm_params()
        agent_one_hot = agent.access_agent_one_hot_id().repeat(1, 32, 1)

        t_agent_observations = tensors["observations"][:, :, agent_id, :]
        t_agent_next_observations = tensors["next_observations"][:, :, agent_id, :]

        t_agent_observations = torch.cat([t_agent_observations, agent_one_hot], dim=2)
        t_agent_next_observations = torch.cat(
            [t_agent_next_observations, agent_one_hot], dim=2
        )

        t_agent_prev_actions = tensors["prev_actions"][:, :, agent_id].unsqueeze(2)
        t_agent_actions = tensors["actions"][:, :, agent_id].unsqueeze(2)

        t_agent_prev_actions_one_hot = torch.nn.functional.one_hot(
            t_agent_prev_actions.squeeze(2), num_classes=14
        )
        t_agent_actions_one_hot = torch.nn.functional.one_hot(
            t_agent_actions.squeeze(2), num_classes=14
        )

        q_vals = agent.estimate_q_values(
            t_agent_observations, t_agent_prev_actions_one_hot
        )
        target_q_vals = agent.estimate_target_q_values(
            t_agent_next_observations, t_agent_actions_one_hot
        )

        agent_avail_actions = tensors["avail_actions"][:, :, agent_id, :].squeeze(0)
        target_q_vals[agent_avail_actions == 0] = -9e7

        return q_vals, target_q_vals

    def _calculate_loss(self, tensors, multi_agent_q_vals, multi_agent_target_q_vals):
        """given target and objective calculate loss w.r.t. criterion"""
        t_multi_agent_q_vals = torch.stack(multi_agent_q_vals, dim=0).transpose(0, 1)
        t_multi_agent_target_q_vals = torch.stack(
            multi_agent_target_q_vals, dim=0
        ).transpose(0, 1)

        prev_actions_taken = tensors["prev_actions"].unsqueeze(-1)
        online_mixing_q_vals = t_multi_agent_q_vals.gather(
            dim=-1, index=prev_actions_taken.squeeze(0)
        )

        q_tot = self._online_mixing_network(online_mixing_q_vals, tensors["states"])
        q_tot = q_tot.squeeze(1)

        max_actions = t_multi_agent_target_q_vals.argmax(dim=-1).unsqueeze(-1)
        target_mixing_q_vals = t_multi_agent_target_q_vals.gather(
            dim=-1, index=max_actions
        )
        target_q_tot = self._target_mixing_network(
            target_mixing_q_vals, tensors["next_states"]
        )
        target_q_tot = target_q_tot.squeeze(1)

        t_rewards_transposed = tensors["rewards"].transpose(0, 1)
        t_dones_transposed = tensors["dones"].transpose(0, 1)
        y_hat = t_rewards_transposed + (
            self._discount_factor * target_q_tot * (1 - t_dones_transposed)
        )
        y_hat = y_hat.detach()

        loss = self._criterion(q_tot, y_hat)
        return loss

    def _backpropagate(self, loss):
        """backprop through network parameters"""
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._params, self._grad_clip)
        self._optimizer.step()

    def update_target_networks(self, n_rollout, n_rollouts_per_target_swap):
        """given schedule update target frozen weights of online network weights"""
        if n_rollout % n_rollouts_per_target_swap == 0:
            for agent in self._agents:
                agent.update_target_network_params()
            self.update_target_network_params()

    def decrease_exploration_rate(self, n_rollout, n_rollouts_per_epsilon_decrease):
        """decrease exploration epsilon param in each agent after n_rollouts_per_epsilon_decrease passed"""
        if n_rollout % n_rollouts_per_epsilon_decrease == 0:
            for agent in self._agents:
                agent.decrease_exploration()

    def optimize(self):
        """opimize network - main learn method"""
        sample_data = self._memory.sample_buffer()
        tensors = self._convert_to_tensors(sample_data)
        multi_agent_q_vals, multi_agent_target_q_vals = self._get_multi_agent_q_values(
            tensors
        )

        loss = self._calculate_loss(
            tensors, multi_agent_q_vals, multi_agent_target_q_vals
        )
        self._backpropagate(loss)

    # ==================================
    # Section 3: Eval
    # ==================================

    def evaluate(self, n_games: int):
        """evaluate current model on n_games"""
        results = []
        for _ in tqdm(range(n_games), desc="Evaluation Phase: "):
            self._initialize_environment()
            terminated = False
            episode_return = 0

            while not terminated:
                observations, _ = self._fetch_environment_details()
                actions, _ = self._get_agent_actions(
                    observations, self._timesteps, evaluate=True
                )
                reward, terminated = self._execute_actions(actions)
                episode_return += reward

            results.append(episode_return)

        mean_score = np.mean(results)
        print("EVAL MEAN SCORE: ", mean_score)
        return mean_score
