from dataclasses import dataclass
from dataclasses import field


@dataclass
class TrainableConfig:
    hypernetwork_embedding_size: int
    hypernetwork_n_layers: int

    mixer_embedding_size: int

    learner_embedding_size: int
    learner_hidden_state_size: int
    learner_lr: float
    learner_gamma: float
    learner_grad_clip: float
    learner_target_net_update_schedule: int
    learner_epsilon_start: float
    learner_epsilon_min: float
    learner_epsilon_anneal_steps: int

    buffer_max_size: int
    buffer_batch_size: int
    buffer_type: str
