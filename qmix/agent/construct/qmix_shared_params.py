import numpy as np
import torch
from agent.abstract.base_construct import BaseConstruct

from qmix.agent.construct.entity import DRQNAgent
from qmix.agent.networks import DRQN
from qmix.agent.networks import MixingNetwork
from qmix.common import constants
from qmix.common import methods
from qmix.environment import SC2Environment


class QMIXSharedParamsConstruct(torch.nn.Module, BaseConstruct):
    def __init__(self, construct_registry_directive: dict):
        super().__init__()
        self._construct_registry_directive = construct_registry_directive
        self._construct_configuration = None

        # Instantiated agents
        self._agents = None

        # Instantiated env
        self._environment = None

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

    def _instantiate_optimizer_and_criterion(self, learning_rate: float):
        self._optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self._criterion = torch.nn.MSELoss()

    def _instantiate_factorisation_network(
        self,
        hypernetwork_configuration: dict,
        mixing_network_configuration: dict,
    ):
        self._online_mixing_network = MixingNetwork(
            mixing_network_configuration=mixing_network_configuration,
            hypernetwork_configuration=hypernetwork_configuration,
        ).construct_network()

        self._target_mixing_network = MixingNetwork(
            mixing_network_configuration=mixing_network_configuration,
            hypernetwork_configuration=hypernetwork_configuration,
        ).construct_network()

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

    def _create_environment_instance(self, environment_configuration: dict):
        self._environment = SC2Environment(environment_configuration)

    def _check_construct(self):
        assert self._agents is not None, "Agents are not spawned"
        assert (
            self._online_mixing_network is not None
        ), "Online mixing network is not instantiated"
        assert (
            self._target_mixing_network is not None
        ), "Target mixing network is not instantiated"
        assert self._environment is not None, "Environment is not instantiated"
        assert self._optimizer is not None, "Optimizer is not instantiated"
        assert self._criterion is not None, "Objective function is not instantiated"

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
        )

        self._spawn_agents(
            drqn_configuration=drqn_configuration,
            num_actions=num_actions,
            num_agents=num_agents,
        )

        env_config = {"env_name": "8m"}
        self._create_environment_instance(environment_configuration=env_config)
        self._instantiate_optimizer_and_criterion(learning_rate=self._learning_rate)
        self._check_construct()

        return self

    def optimize(self, batch: np.ndarray):
        pass
