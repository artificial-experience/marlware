from typing import Tuple

from omegaconf import DictConfig
from omegaconf import OmegaConf

from src.node import container


def fill_trial_config(conf: DictConfig) -> OmegaConf:
    """structure omegaconf and fill the necessary dataclasses"""
    train_conf = conf["rollout"]["train"]
    test_conf = conf["rollout"]["test"]
    combined_dict = {**train_conf, **test_conf}
    rollout_dict_conf = OmegaConf.create(combined_dict)

    device_dict_conf = conf["device"]

    runtime_config = container.RuntimeConfig(**rollout_dict_conf)
    device_config = container.DeviceConfig(**device_dict_conf)

    structured_conf = OmegaConf.structured(
        container.TrialConfig(
            runtime=runtime_config,
            device=device_config,
        )
    )
    return structured_conf


def fill_trainable_config(conf: DictConfig) -> OmegaConf:
    """structure omegaconf and fill the necessary dataclasses"""
    trainable_dict_config = conf["trainable"]

    trainable_conf = container.ConstructConfig(
        construct=container.ConstructImplementationConfig(
            **trainable_dict_config["trainable"]["construct"]
        ),
        hypernetwork=container.ConstructHypernetworkModelConfig(
            **trainable_dict_config["trainable"]["hypernetwork"]["model"]
        ),
        mixer=container.ConstructMixerModelConfig(
            **trainable_dict_config["trainable"]["mixer"]["model"]
        ),
    )

    learner_conf = container.LearnerConfig(
        training=container.LearnerTrainingConfig(
            **trainable_dict_config["learner"]["training"]
        ),
        model=container.LearnerModelConfig(**trainable_dict_config["learner"]["model"]),
        exploration=container.LearnerExplorationConfig(
            **trainable_dict_config["learner"]["exploration"]
        ),
    )

    buffer_conf = container.BufferConfig(**trainable_dict_config["buffer"])

    structured_conf = OmegaConf.structured(
        container.TrainableConfig(
            trainable=trainable_conf,
            learner=learner_conf,
            buffer=buffer_conf,
        )
    )
    return structured_conf


def deserialize_configuration_node(
    cfg: DictConfig,
) -> Tuple[container.TrainableConfig, container.TrialConfig]:
    """Serialize a DictConfig node to a YAML string."""
    trainable_conf = fill_trainable_config(cfg)
    trial_conf = fill_trial_config(cfg)
    return trainable_conf, trial_conf
