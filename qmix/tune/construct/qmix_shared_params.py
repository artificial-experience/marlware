import numpy as np
import torch

from qmix.abstract.construct import BaseConstruct
from qmix.agent import DRQNAgent
from qmix.common import constants
from qmix.common import methods
from qmix.environment import SC2Environment
from qmix.memory import initialize_memory
from qmix.networks import DRQN
from qmix.networks import MixingNetwork


class QMIXSharedParamsConstruct(BaseConstruct):
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

        self._online_mixing_network = None
        self._target_mixing_network = None

        # Intrinsic const parameters
        self._learning_rate = None
        self._discount_factor = None
        self._target_network_update_schedule = None

        # Default device set to cpu
        self._accelerator_device = "cpu"

        # Construct optimizer and criterion
        self._optimizer = None
        self._criterion = None
        self._optimizer_step = 0

    @classmethod
    def from_construct_registry_directive(cls, construct_registry_directive: str):
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
        instance._accelerator_device = methods.get_nested_dict_field(
            directive=configuration,
            keys=[
                "architecture-directive",
                "device_configuration",
                "accelerator",
                "choice",
            ],
        )
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
        return instance

    def _instantiate_optimizer_and_criterion(self, parameters, learning_rate: float):
        self._optimizer = torch.optim.Adam(params=parameters, lr=learning_rate)
        self._criterion = torch.nn.MSELoss()

    def _instantiate_factorisation_network(
        self,
        hypernetwork_configuration: dict,
        mixing_network_configuration: dict,
        num_agents: int,
    ):
        self._online_mixing_network = MixingNetwork(
            mixing_network_configuration=mixing_network_configuration,
            hypernetwork_configuration=hypernetwork_configuration,
        ).construct_network(num_agents=num_agents)

        self._target_mixing_network = MixingNetwork(
            mixing_network_configuration=mixing_network_configuration,
            hypernetwork_configuration=hypernetwork_configuration,
        ).construct_network(num_agents=num_agents)

    def _spawn_agents(
        self, drqn_configuration: dict, num_actions: int, num_agents: int
    ):
        # Create target and online networks
        self._shared_target_drqn_network = DRQN(
            config=drqn_configuration
        ).construct_network()
        self._shared_online_drqn_network = DRQN(
            config=drqn_configuration
        ).construct_network()

        self._agents = [
            DRQNAgent(
                agent_unique_id=identifier,
                agent_configuration=drqn_configuration,
                num_actions=num_actions,
                target_drqn_network=self._shared_target_drqn_network,
                online_drqn_network=self._shared_online_drqn_network,
            )
            for identifier in range(num_agents)
        ]

    def _instantiate_env(self, environment_configuration: dict):
        environment_creator = SC2Environment(config=environment_configuration)
        (
            self._environment,
            self._environment_info,
        ) = environment_creator.create_env_instance()

    def _instantiate_replay_memory(self, memory_configuration: dict):
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

        states_field = "states"
        states_vals = np.zeros([max_size, 168], dtype=np.float32)

        extra_fields = (prev_actions_field, states_field)
        extra_vals = (prev_actions_vals, states_vals)

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
        assert self._agents is not None, "Agents are not spawned"
        assert (
            self._online_mixing_network is not None
        ), "Online mixing network is not instantiated"
        assert (
            self._target_mixing_network is not None
        ), "Target mixing network is not instantiated"
        assert self._environment is not None, "Environment is not instantiated"

    def commit(self):
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
            keys=["architecture-directive", "mixing_network_configuration"],
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
        self._check_construct()

        return self

    def _access_construct_parameters(self):
        parameters = []

        parameters += list(self._shared_target_drqn_network.parameters())
        parameters += list(self._shared_online_drqn_network.parameters())
        parameters += list(self._online_mixing_network.parameters())
        parameters += list(self._target_mixing_network.parameters())

        return parameters

    # TODO: break down into smaller chunks
    def optimize(self, n_rollouts: int, steps_per_rollout_limit: int):
        construct_params = self._access_construct_parameters()
        self._instantiate_optimizer_and_criterion(
            parameters=construct_params, learning_rate=self._learning_rate
        )

        episode_scores = []
        for rollout in range(n_rollouts):
            if self._memory.ready():
                sample = self._memory.sample_buffer()

                # Train the network

            self._environment.reset()
            terminated = False
            episode_return = 0
            prev_actions = None

            while not terminated:
                observations = self._environment.get_obs()
                states = self._environment.get_state()

                actions = []
                num_agents = len(self._agents)
                for agent_id in range(num_agents):
                    available_actions = self._environment.get_avail_agent_actions(
                        agent_id
                    )
                    available_actions_index = np.nonzero(available_actions)[0]

                    agent = self._agents[agent_id]
                    assert agent_id == agent._agent_unique_id

                    agent_action = agent.act(observations)
                    if agent_action not in available_actions_index:
                        agent_action = np.random.choice(available_actions_index)

                    actions.append(agent_action)

                reward, terminated, _ = self._environment.step(actions)
                episode_return += reward

                next_observations = self._environment.get_obs()

                if prev_actions is not None:
                    data = [
                        observations,
                        actions,
                        reward,
                        next_observations,
                        terminated,
                        prev_actions,
                        states,
                    ]
                    self._memory.store_transition(data)

                prev_actions = actions

            episode_scores.append(episode_return)

        self._environment.close()

        return episode_return
