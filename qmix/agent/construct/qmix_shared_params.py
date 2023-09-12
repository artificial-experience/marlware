from agent.abstract.base_construct import BaseConstruct

from qmix.agent.networks import DRQN
from qmix.agent.networks import HyperNetwork
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
        self._drqn_entity = None
        self._hypernetwork_entity = None
        self._mixing_network_entity = None

    @classmethod
    def from_construct_registry_directive(cls, construct_registry_directive: str):
        instance = cls(construct_registry_directive)
        path_to_construct_file = construct_registry_directive.get(
            "path_to_construct_file", None
        )
        construct_file_path = (
            constants.Directories.TRAINABLE_CONFIG_DIR.value / path_to_construct_file
        )
        instance._construct_configuration = methods.load_yaml(
            construct_file_path.absolute()
        )
        instance._num_agents = construct_registry_directive.get("num_agents", None)
        return instance

    def _instantiate_construct(
        self,
        drqn_configuration: dict,
        hypernetwork_configuration: dict,
        mixing_network_configuration: dict,
    ):
        self._drqn_entity = DRQN(config=drqn_configuration).construct_network()
        self._hypernetwork_entity = HyperNetwork(
            config=hypernetwork_configuration
        ).construct_network()
        self._mixing_network_entity = MixingNetwork(
            config=mixing_network_configuration
        ).construct_network()

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
        self._instantiate_construct(
            drqn_configuration=drqn_configuration,
            hypernetwork_configuration=hypernetwork_configuration,
            mixing_network_configuration=mixing_network_configuration,
        )
        return self
