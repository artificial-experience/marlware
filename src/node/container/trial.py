from dataclasses import dataclass
from dataclasses import field


@dataclass
class RuntimeConfig:
    n_timesteps: int
    eval_schedule: int
    checkpoint_frequency: int
    n_games: int
    display_freq: int
    warmup: int


@dataclass
class DeviceConfig:
    num_workers: int
    accelerator: str
    seed: int


@dataclass
class TrialConfig:
    runtime: RuntimeConfig
    device: DeviceConfig
