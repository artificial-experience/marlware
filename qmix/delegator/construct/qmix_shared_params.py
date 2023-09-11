from delegator.abstract.base_construct import BaseConstruct

from qmix import environment
from qmix.common import constants
from qmix.common import methods


class QMIXSharedParamsConstruct(BaseConstruct):
    def __init__(self, construct_registry_directive: dict):
        self._construct_registry_directive = construct_registry_directive
        self._construct_configuration = None

        # Networks
        self._mixing_network_entity = None
        self._drqn_entity = None
        self._hypernetwork_entity = None

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
        return instance

    def _instantiate_construct(
        self,
        drqn_configuration: dict,
        hyper_network_configuration: dict,
        mixing_network_configuration: dict,
    ):
        pass

    def commit(self):
        print(self._construct_configuration)
