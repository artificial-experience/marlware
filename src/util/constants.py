import os
from enum import Enum
from pathlib import Path


ROOT_DIR = Path(os.getenv("ROOT", ".")) / "src"
RESULTS_DIR = ROOT_DIR / "results"
CONF_DIR = ROOT_DIR / "conf"
WEIGHTS_DIR = ROOT_DIR / "weights"
TRAINABLE_CONF_DIR = CONF_DIR / "trainable"


class _EnvKey(Enum):
    """configuration attribute keys for environment"""

    _STATE_SHAPE = "state_shape"
    _OBS_SHAPE = "obs_shape"
    _N_AGENTS = "n_agents"
    _N_ACTIONS = "n_actions"
    _EP_LIMIT = "episode_limit"
    _BATTLE_WON = "battle_won"


class _MemoryKey(Enum):
    """configuraiton attribute keys for memory replay"""

    _SCHEME = "scheme"
    _GROUP = "groups"
    _VALUE_SHAPE = "vshape"
    _DTYPE_KEY = "dtype"
    _EP_CONST = "episode_const"

    # groups and transforms
    _AGENT_GROUP = "agents"
    _ACTIONS_ONEHOT_TRANSFORM = "actions_onehot"

    # episode store
    _STATE = "state"
    _OBS = "observation"
    _ACTIONS = "actions"
    _AVAIL_ACTIONS = "avail_actions"
    _PROBS = "probs"
    _REWARD = "reward"
    _TERMINATED = "terminated"


class _CortexKey(Enum):
    """configuraiton attribute keys for cortex"""

    pass


class _TunerKey(Enum):
    """configuraiton attribute keys for tuner"""

    pass


class _LoggerKey(Enum):
    """configuration attribute keys for logger"""


class AttrKey:
    """attribute key manager"""

    # hold configuration keys for components
    env = _EnvKey
    memory = _MemoryKey
    cortex = _CortexKey
    tuner = _TunerKey
    logger = _LoggerKey

    def get_attr_keys(self, attr: Enum) -> str:
        """return all registered key-attributes"""
        pass
