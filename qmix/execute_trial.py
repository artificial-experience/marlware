from pathlib import Path

import click
from common import constants
from common import methods
from tune import Tuner


class Tune:
    """
    Interface class between tuner and trainable

    Args:
        :param [config]: configuration dictionary for engine

    Internal State:
        :param [trainable]: algorithm and objective function to be trained on the problem
    """

    def __init__(self, config_name: dict):
        self._config_name = config_name

        self._trial_configuration = None
        self._construct_directive = None
        self._tuner_directive = None

        self._tuner = None

        self._results = None
        self._logger = None

        # TODO connect W&B
        self._monitor = None

    def _set_trial_configuration(self):
        config_directory = constants.Directories.CONFIG_DIR.value / self._config_name
        self._trial_configuration = methods.load_yaml(config_directory.absolute())
        self._construct_directive = self._trial_configuration.get("construct-directive")
        self._tuner_directive = self._trial_configuration.get("tuner-directive")

    def _prepare_tuner(self):
        self._tuner = Tuner.from_trial_directive(
            construct_directive=self._construct_directive,
            tuner_directive=self._tuner_directive,
        )

    # TODO: add possibility to connect to AWS as remote=True
    def execute_trial(self, remote=False):
        self._set_trial_configuration()
        self._prepare_tuner()

        results = self._tuner.fit()

        if results:
            # make some logging and stuff
            self._results = results
        else:
            pass


if __name__ == "__main__":
    trial_directives = "qmix-trial-directives.yaml"
    tune = Tune(trial_directives)
    tune.execute_trial()
