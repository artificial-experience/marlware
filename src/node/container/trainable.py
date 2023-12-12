from dataclasses import dataclass


@dataclass
class ConstructImplementationConfig:
    impl: str


@dataclass
class ConstructHypernetworkModelConfig:
    embedding_dim: int
    n_layers: int


@dataclass
class ConstructMixerModelConfig:
    embedding_dim: int


@dataclass
class LearnerTrainingConfig:
    lr: float
    gamma: float
    grad_clip: float
    target_net_update_shedule: int


@dataclass
class LearnerModelConfig:
    rnn_hidden_dim: int


@dataclass
class LearnerExplorationConfig:
    epsilon_start: float
    epsilon_min: float
    epsilon_anneal_steps: int


@dataclass
class ConstructConfig:
    construct: ConstructImplementationConfig
    hypernetwork: ConstructHypernetworkModelConfig
    mixer: ConstructMixerModelConfig


@dataclass
class LearnerConfig:
    training: LearnerTrainingConfig
    model: LearnerModelConfig
    exploration: LearnerExplorationConfig


@dataclass
class BufferConfig:
    mem_size: int
    batch_size: int
    prioritized: bool
    mode: str


@dataclass
class TrainableConfig:
    trainable: ConstructConfig
    learner: LearnerConfig
    buffer: BufferConfig
