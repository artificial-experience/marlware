from dataclasses import dataclass
from dataclasses import field


@dataclass
class TrialConfig:
    n_rollouts: int
    eval_schedule: int
    checkpoint_frequency: int
    n_games: int
    device_accelerator: str
