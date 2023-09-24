import torch
from torchviz import make_dot

from .trainable import TrainableConstruct


class Tuner:
    def __init__(self, construct_directive: dict, tuner_directive: dict):
        self.construct_directive = construct_directive
        self.tuner_directive = tuner_directive

        self._trainable_construct = None

    @classmethod
    def from_trial_directive(cls, construct_directive: dict, tuner_directive: dict):
        instance = cls(construct_directive, tuner_directive)
        instance._trainable_construct = (
            TrainableConstruct.from_construct_directive(
                construct_directive=construct_directive
            )
        ).delegate()
        return instance

    def fit(self):
        torch.autograd.set_detect_anomaly(True)
        self._trainable_construct.optimize(
            n_rollouts=15000, steps_per_rollout_limit=120
        )
