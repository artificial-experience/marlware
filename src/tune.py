import json
import logging
from logging import Logger
from typing import Tuple

import hydra
import wandb
from logger import TraceLogger
from node import deserialize_configuration_node
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tuner import CoreTuner


def delegate_tuner(
    environ_prefix: str,
    configuration: OmegaConf,
    accelerator: str,
    trace_logger: Logger,
    *,
    seed: int
) -> CoreTuner:
    """delegate tuner w.r.t passed configuration"""
    tuner = CoreTuner(configuration)
    tuner.commit(
        environ_prefix=environ_prefix,
        accelerator=accelerator,
        logger=trace_logger,
        seed=seed,
    )
    return tuner


def access_trial_directives(configuration: OmegaConf) -> Tuple[OmegaConf, OmegaConf]:
    """access configuration and get runtime and device settings"""
    runtime = configuration.get("runtime", None)
    device = configuration.get("device", None)
    return runtime, device


def get_logger() -> logging.Logger:
    """get hydra logger"""
    log = logging.getLogger(__name__)
    return log


def format_config_file(cfg: OmegaConf) -> str:
    """reformat to yaml for a nice display and add a message"""
    message = "Starting Trial with Hydra Configuration ...\n"
    formatted_config = OmegaConf.to_yaml(cfg, resolve=True)
    return message + formatted_config


def start_wandb(cfg: DictConfig):
    """create wandb instance"""
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    return run


@hydra.main(version_base=None, config_path="conf", config_name="trial")
def runner(cfg: DictConfig) -> None:
    """execute trial"""
    # TODO: add wandb support
    # run = start_wandb(cfg)

    logger = get_logger()
    formatted_config = format_config_file(cfg)
    logger.info(formatted_config)
    trace_logger = TraceLogger(logger)

    trainable_conf, trial_conf = deserialize_configuration_node(cfg)
    runtime, device = access_trial_directives(trial_conf)

    accelerator = device.get("accelerator", "cpu")
    seed = device.get("seed", None)
    tuner = delegate_tuner("3s5z", trainable_conf, accelerator, trace_logger, seed=seed)

    n_rollouts = runtime.n_rollouts
    eval_schedule = runtime.eval_schedule
    checkpoint_freq = runtime.checkpoint_frequency
    eval_n_games = runtime.n_games

    tuner.optimize(n_rollouts, eval_schedule, checkpoint_freq, eval_n_games)


if __name__ == "__main__":
    runner()
