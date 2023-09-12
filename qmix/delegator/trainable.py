from common import methods

from .abstract.registration import ConstructRegistry
from .construct.qmix_shared_params import QMIXSharedParamsConstruct


class TrainableConstructDelegator:
    def __init__(self, construct_directive: dict):
        self._construct_directive = construct_directive
        print(self._construct_directive)

        self._construct_configuration_file = None
        self._registered_trainable_constructs = None
        self._target_trainable_construct = None
        self._num_agents = None

    @classmethod
    def from_construct_directive(cls, construct_directive: dict):
        instance = cls(construct_directive)
        instance._construct_configuration_file = methods.get_nested_dict_field(
            directive=construct_directive,
            keys=["trainable_configuration", "config_name", "choice"],
        )
        instance._target_trainable_construct = methods.get_nested_dict_field(
            directive=construct_directive,
            keys=["trainable_configuration", "construct_class", "choice"],
        )
        instance._num_agents = methods.get_nested_dict_field(
            directive=construct_directive,
            keys=["trainable_configuration", "num_agents", "choice"],
        )
        instance._registered_trainable_constructs = (
            ConstructRegistry.get_registered_constructs()
        )
        return instance

    def delegate(self):
        construct = None
        if self._target_trainable_construct in self._registered_trainable_constructs:
            construct = ConstructRegistry.create(
                target_construct_class=self._target_trainable_construct,
                path_to_construct_file=self._construct_configuration_file,
                num_agents=self._num_agents,
            ).commit()
        else:
            raise SystemError(
                f"ERROR: {self._target_trainable_construct} is not registered"
            )
        return construct
