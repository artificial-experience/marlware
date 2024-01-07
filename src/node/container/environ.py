from dataclasses import dataclass
from dataclasses import field
from typing import Optional


@dataclass
class EnvArgs:
    map_name: str
    continuing_episode: bool
    difficulty: str
    game_version: Optional[str]
    move_amount: int
    obs_all_health: bool
    obs_instead_of_state: bool
    obs_last_action: bool
    obs_own_health: bool
    obs_pathing_grid: bool
    obs_terrain_height: bool
    obs_timestep_number: bool
    reward_death_value: int
    reward_defeat: int
    reward_negative_scale: float
    reward_only_positive: bool
    reward_scale: bool
    reward_scale_rate: int
    reward_sparse: bool
    reward_win: int
    conic_fov: bool
    use_unit_ranges: bool
    min_attack_range: int
    obs_own_pos: bool
    num_fov_actions: int
    fully_observable: bool

    state_last_action: bool
    state_timestep_number: bool
    step_mul: int
    heuristic_ai: bool
    debug: bool
    prob_obs_enemy: float
    action_mask: bool

    # smacv2 extenstion
    capability_config: dict

    window_size_x: int
    window_size_y: int


@dataclass
class EnvironConfig:
    args: EnvArgs
