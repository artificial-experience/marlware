from typing import Tuple

import click
import hydra
from node import serialize_configuration_node
from omegaconf import DictConfig
from omegaconf import OmegaConf
from trainable import TrainableConstruct


def delegate_trainable(
    environ_prefix: str, configuration: OmegaConf, accelerator: str, seed: int
) -> TrainableConstruct:
    """delegate trainable w.r.t passed configuration"""
    construct = TrainableConstruct(configuration)
    construct.commit(environ_prefix=environ_prefix, accelerator=accelerator, seed=seed)
    return construct


def access_trial_directives(configuration: OmegaConf) -> Tuple[OmegaConf, OmegaConf]:
    """access configuration and get runtime and device settings"""
    runtime = configuration.get("runtime", None)
    device = configuration.get("device", None)
    return runtime, device


@click.command()
@click.option("--config_name", default="trial", help="Name of the Hydra config file.")
@click.option(
    "--environ_prefix", default="8m", help="Name of the sc2 environ config file."
)
def tune(config_name: str, environ_prefix: str):
    """accept input parameters"""

    @hydra.main(version_base=None, config_path="conf", config_name=config_name)
    def hydra_main(cfg: DictConfig) -> None:
        """execute trial"""
        trainable_conf, trial_conf = serialize_configuration_node(cfg)
        runtime, device = access_trial_directives(trial_conf)

        accelerator = device.get("accelerator", "cpu")
        seed = device.get("seed", None)
        trainable = delegate_trainable(
            environ_prefix, trainable_conf, accelerator, seed=seed
        )

    hydra_main()


if __name__ == "__main__":
    tune()
