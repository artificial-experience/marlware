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

        # Number of agents in MARL setting
        self._num_agents = None

        # Default device set to cpu
        self._accelerator_device = "cpu"

        # Networks
        self._drqn_agent = None
        self._mixing_network_entity = None

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
            keys=["device_configuration", "accelerator", "choice"],
        )
        instance._num_agents = methods.get_nested_dict_field(
            directive=configuration,
            keys=["environment-directive", "num_agents", "choice"],
        )
        return instance

    def _instantiate_factorisation_network(
        self,
        hypernetwork_configuration: dict,
        mixing_network_configuration: dict,
    ):
        self._mixing_network_entity = MixingNetwork(
            mixing_network_configuration=mixing_network_configuration,
            hypernetwork_configuration=mixing_network_configuration,
        ).construct_network()

    def _spawn_agents(self, drqn_configuration):
        self._drqn_agent = DRQN(config=drqn_configuration).construct_network()

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
        self._instantiate_factorisation_network(
            hypernetwork_configuration=hypernetwork_configuration,
            mixing_network_configuration=mixing_network_configuration,
        )
        self._spawn_agents(drqn_configuration=drqn_configuration)
        return self
