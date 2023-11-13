from typing import Tuple

import hydra
from node import deserialize_configuration_node
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tuner import ProtoTuner


def delegate_tuner(
    environ_prefix: str, configuration: OmegaConf, accelerator: str, seed: int
) -> ProtoTuner:
    """delegate tuner w.r.t passed configuration"""
    tuner = ProtoTuner(configuration)
    tuner.commit(environ_prefix=environ_prefix, accelerator=accelerator, seed=seed)
    return tuner


def access_trial_directives(configuration: OmegaConf) -> Tuple[OmegaConf, OmegaConf]:
    """access configuration and get runtime and device settings"""
    runtime = configuration.get("runtime", None)
    device = configuration.get("device", None)
    return runtime, device


@hydra.main(version_base=None, config_path="conf", config_name="trial")
def runner(cfg: DictConfig) -> None:
    """execute trial"""
    trainable_conf, trial_conf = deserialize_configuration_node(cfg)
    runtime, device = access_trial_directives(trial_conf)

    accelerator = device.get("accelerator", "cpu")
    seed = device.get("seed", None)
    tuner = delegate_tuner("3m", trainable_conf, accelerator, seed=seed)

    n_rollouts = runtime.n_rollouts
    eval_schedule = runtime.eval_schedule
    checkpoint_freq = runtime.checkpoint_frequency
    eval_n_games = runtime.n_games

    score = tuner.optimize(n_rollouts, eval_schedule, checkpoint_freq, eval_n_games)


if __name__ == "__main__":
    runner()
