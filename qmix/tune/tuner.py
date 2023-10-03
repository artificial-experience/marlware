import torch
from torchviz import make_dot

from .trainable import TrainableConstruct


class Tuner:
    """
    Interface class between tuner and trial execution

    Args:
        :param [config_name]: configuration name for the trial

    Internal State:
        :param [construct_directive]: directive to be used once initialising construct class
        :param [tuner_directive]: directive to be used once initialising tuner class
        :param [trainable_construct]: trainable instance to be optimized w.r.t objective function
    """

    def __init__(self, construct_directive: dict, tuner_directive: dict):
        self.construct_directive = construct_directive
        self.tuner_directive = tuner_directive

        self._trainable_construct = None

    @classmethod
    def from_trial_directive(cls, construct_directive: dict, tuner_directive: dict):
        """Instantiate trainable construct from directive"""
        instance = cls(construct_directive, tuner_directive)
        instance._trainable_construct = (
            TrainableConstruct.from_construct_directive(
                construct_directive=construct_directive
            )
        ).delegate()
        return instance

    def fit(self):
        """Start rollout and optimize trainable construct"""
        torch.autograd.set_detect_anomaly(True)
        self._trainable_construct.optimize(
            n_rollouts=40000, steps_per_rollout_limit=120
        )
