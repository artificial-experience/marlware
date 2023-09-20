import torch
from torchviz import make_dot

from .trainable import TrainableConstructDelegator


class AgentTunerDelegator:
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
        print(trainable_construct)
        q_values = torch.rand(112)
        state = torch.rand(160)
        y = trainable_construct(q_values, state)
        print(y)
        make_dot(
            y.mean(),
            params=dict(trainable_construct.named_parameters()),
            show_attrs=True,
            show_saved=True,
        ).render("qmix_network", format="png")
        tuner = None

        return tuner

    # TODO: Implement separate tuner abstraction class and move this method outside of this class
    def start_rollout(self):
        pass
