from common import methods

from .trainable import TrainableConstructDelegator


class TunerDelegator:
    def __init__(self, construct_directive: dict, tuner_directive: dict):
        self.construct_directive = construct_directive
        self.tuner_directive = tuner_directive

        self._trainable_construct_delegator = None

    @classmethod
    def from_trial_directive(cls, construct_directive: dict, tuner_directive: dict):
        instance = cls(construct_directive, tuner_directive)
        instance._trainable_construct_delegator = (
            TrainableConstructDelegator.from_construct_directive(
                construct_directive=construct_directive
            )
        )
        return instance

    def delegate_tuner_entity(self):
        """Instantiate and return tuner entity ready for training"""
        trainable_construct = self._trainable_construct_delegator.delegate()
        tuner = None

        return tuner

    # TODO: Implement separate tuner abstraction class and move this method outside of this class
    def start_rollout(self):
        pass
