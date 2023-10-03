from pathlib import Path

import click
from common import constants
from common import methods
from tune import Tuner


class Tune:
    """
    Interface class between tuner and trial execution

    Args:
        :param [config_name]: configuration name for the trial

    Internal State:
        :param [trial_configuration]: contents of the config file
        :param [construct_directive]: class and networks to be used int trainable
        :param [tuner_directive]: rollout and checkpoint configuration
        :param [tuner]: entity of tuner to be executed
        :param [results]: results produced by the tuner
        :param [logger]: logger instance producing output messages
        :param [monitor]: external monitoring system
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
        """Read configuration and set internal vars"""
        config_directory = constants.Directories.CONFIG_DIR.value / self._config_name
        self._trial_configuration = methods.load_yaml(config_directory.absolute())
        self._construct_directive = self._trial_configuration.get("construct-directive")
        self._tuner_directive = self._trial_configuration.get("tuner-directive")

    def _prepare_tuner(self):
        """Instantiate tuner instance"""
        self._tuner = Tuner.from_trial_directive(
            construct_directive=self._construct_directive,
            tuner_directive=self._tuner_directive,
        )

    # TODO: add possibility to connect to AWS as remote=True
    def execute_trial(self, remote=False):
        """Prepare trial and tuner and run"""

        self._set_trial_configuration()
        self._prepare_tuner()

        results = self._tuner.fit()

        if results:
            # make some logging and stuff
            self._results = results
        else:
            pass

    def access_trial_results(self):
        return self._results


if __name__ == "__main__":
    trial_directives = "qmix-trial-directives.yaml"
    tune = Tune(trial_directives)
    tune.execute_trial()
