import os
from enum import Enum
from pathlib import Path


ROOT_DIR = Path(os.getenv("ROOT", "."))
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = ROOT_DIR / "outputs"
REPLAY_DIR = RESULTS_DIR / "replays"
MODEL_SAVE_DIR = RESULTS_DIR / "models"
CONF_DIR = SRC_DIR / "conf"
WEIGHTS_DIR = SRC_DIR / "weights"
TRAINABLE_CONF_DIR = CONF_DIR / "trainable"


class _EnvKey(Enum):
    """configuration attribute keys for environment"""

    _STATE_SHAPE = "state_shape"
    _OBS_SHAPE = "obs_shape"
    _N_AGENTS = "n_agents"
    _N_ACTIONS = "n_actions"
    _EP_LIMIT = "episode_limit"
    _BATTLE_WON = "battle_won"


class _DataKey(Enum):
    """configuraiton attribute keys for data replay"""

    _SCHEME = "scheme"
    _GROUP = "group"
    _VALUE_SHAPE = "vshape"
    _DTYPE = "dtype"
    _MAX_SEQ_LEN = "_max_seq_length"
    _TRANSFORMS = "transforms"

    # groups and transforms
    _AGENT_GROUP = "agents"
    _ACTIONS_ONEHOT_TRANSFORM = "actions_onehot"

    # episode store
    _STATE = "state"
    _OBS = "obs"
    _ACTIONS = "actions"
    _AVAIL_ACTIONS = "avail_actions"
    _PROBS = "probs"
    _REWARD = "reward"
    _TERMINATED = "terminated"
    _FILLED = "filled"


class _CortexKey(Enum):
    """configuraiton attribute keys for cortex"""

    pass


class _TunerKey(Enum):
    """configuraiton attribute keys for tuner"""

    _BATCH_SIZE = "batch_size"
    _MEM_SIZE = "mem_size"


class _LoggerKey(Enum):
    """configuration attribute keys for logger"""


class AttrKey:
    """attribute key manager"""

    # hold configuration keys for components
    env = _EnvKey
    data = _DataKey
    cortex = _CortexKey
    tuner = _TunerKey
    logger = _LoggerKey

    def get_attr_keys(self, attr: Enum) -> str:
        """return all registered key-attributes"""
        pass
