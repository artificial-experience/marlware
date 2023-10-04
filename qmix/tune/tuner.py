import torch
from tqdm import tqdm

from .trainable import TrainableConstruct
from qmix.common import methods


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

        self._n_rollouts = None
        self._eval_schedule = None
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

    def _set_rollout_n_eval_params(self):
        n_rollouts = methods.get_nested_dict_field(
            directive=self.tuner_directive,
            keys=["rollout_configuration", "n_rollouts", "choice"],
        )
        eval_schedule = methods.get_nested_dict_field(
            directive=self.tuner_directive,
            keys=["rollout_configuration", "eval_schedule", "choice"],
        )
        self._n_rollouts = n_rollouts
        self._eval_schedule = eval_schedule

    def fit(self):
        """Start rollout and optimize trainable construct"""
        torch.autograd.set_detect_anomaly(True)
        self._set_rollout_n_eval_params()

        mean_results = []
        for n_rollout in tqdm(range(self._n_rollouts), desc="Training Phase: "):
            self._trainable_construct.update_target_networks(
                n_rollout, n_rollouts_per_target_swap=200
            )

            if n_rollout % self._eval_schedule == 0:
                mean_result = self._trainable_construct.evaluate(n_games=20)
                mean_results.append(mean_result)

            if self._trainable_construct.memory_ready():
                self._trainable_construct.optimize(n_rollout)

            self._trainable_construct.collect_rollouts()

        self._trainable_construct.close_env()
        return mean_results, (self._n_rollouts // self._eval_schedule)
