from agent.abstract.base_construct import BaseConstruct

from qmix.agent.construct.entity import DRQNAgent
from qmix.agent.networks import DRQN
from qmix.agent.networks import MixingNetwork
from qmix.common import constants
from qmix.common import methods
from qmix.environment import SC2Environment


class QMIXSharedParamsConstruct(BaseConstruct):
    def __init__(self, construct_registry_directive: dict):
        self._construct_registry_directive = construct_registry_directive
        self._construct_configuration = None

        # Instantiated agents
        self._agents = None

        # Instantiated env
        self._environment = None

        # Networks
        self._shared_target_drqn_network = None
        self._shared_online_drqn_network = None
        self._mixing_network = None

        # Default device set to cpu
        self.accelerator_device = "cpu"

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
        return instance

    def _instantiate_factorisation_network(
        self,
        hypernetwork_configuration: dict,
        mixing_network_configuration: dict,
    ):
        self._mixing_network = MixingNetwork(
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
                agent_configuration=drqn_configuration,
                num_actions=num_actions,
                target_drqn_network=self._shared_target_drqn_network,
                online_drqn_network=self._shared_online_drqn_network,
            )
            for _ in range(num_agents)
        ]

    def _create_environment_instance(self, environment_configuration: dict):
        self._environment = SC2Environment(environment_configuration)

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

        assert self._agents is not None, "Agents are not spawned"
        assert self._mixing_network is not None, "Mixing network is not instantiated"
        assert self._environment is not None, "Environment is not instantiated"

        return self
