from collections import namedtuple
from datetime import datetime
from functools import reduce
from functools import wraps
from os.path import expandvars
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.nn import functional as F


def load_yaml(yaml_path: str):
    """self explanatory method"""

    def process_dict(dict_to_process):
        for key, item in dict_to_process.items():
            if isinstance(item, dict):
                dict_to_process[key] = process_dict(item)
            elif isinstance(item, str):
                dict_to_process[key] = expandvars(item)
            elif isinstance(item, list):
                dict_to_process[key] = process_list(item)
        return dict_to_process

    def process_list(list_to_process: List):
        """closure for processing yaml list"""
        new_list = []
        for item in list_to_process:
            if isinstance(item, Dict):
                new_list.append(process_dict(item))
            elif isinstance(item, str):
                new_list.append(expandvars(item))
            elif isinstance(item, List):
                new_list.append(process_list(item))
            else:
                new_list.append(item)
        return new_list

    with open(yaml_path) as yaml_file:
        yaml_content = yaml.safe_load(yaml_file)

    return process_dict(yaml_content)


def ensemble_learners(n_agents: int, impl: torch.nn.Module, conf: OmegaConf):
    """prepare one hot encoding and create namedtuple consisting of N learners"""
    Learners = namedtuple("Learners", ["agent" + str(i) for i in range(n_agents)])
    one_hot_configs = [
        F.one_hot(torch.tensor(i), num_classes=n_agents) for i in range(n_agents)
    ]
    learners = Learners(*[impl(conf, one_hot_configs[i]) for i in range(n_agents)])
    return learners


def convert_agent_actions_to_one_hot(
    actions: np.ndarray, n_actions: int
) -> torch.Tensor:
    """convert agents actions into one-hot representation"""
    return F.one_hot(actions, num_classes=n_actions).to(torch.int64)


def get_current_timestamp(use_hour=True):
    """self explanatory method"""
    if use_hour:
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        return datetime.now().strftime("%Y%m%d")


def plot_learning_curve(x, scores, timestamp):
    """given rewards from trial plot running mean"""
    figure_file = f"network-progress-{timestamp}"
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Evaluation Running Average")
    plt.savefig(figure_file)
