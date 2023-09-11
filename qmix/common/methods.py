from datetime import datetime
from functools import reduce
from functools import wraps
from os.path import expandvars
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import yaml


def load_yaml(yaml_path):
    def process_dict(dict_to_process):
        for key, item in dict_to_process.items():
            if isinstance(item, dict):
                dict_to_process[key] = process_dict(item)
            elif isinstance(item, str):
                dict_to_process[key] = expandvars(item)
            elif isinstance(item, list):
                dict_to_process[key] = process_list(item)
        return dict_to_process

    def process_list(list_to_process):
        new_list = []
        for item in list_to_process:
            if isinstance(item, dict):
                new_list.append(process_dict(item))
            elif isinstance(item, str):
                new_list.append(expandvars(item))
            elif isinstance(item, list):
                new_list.append(process_list(item))
            else:
                new_list.append(item)
        return new_list

    with open(yaml_path) as yaml_file:
        yaml_content = yaml.safe_load(yaml_file)

    return process_dict(yaml_content)


def get_current_timestamp(use_hour=True):
    if use_hour:
        return datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        return datetime.now().strftime("%Y%m%d")


def get_nested_dict_field(
    *, directive: Dict[str, Any], keys: List[str]
) -> Optional[Any]:
    """
    Get a nested value from a dictionary.

    Args:
        directives: The target dictionary.
        keys: A list of keys representing the path to the target value in the dictionary.

    Returns:
        The target value if it exists in the dictionary. Otherwise, returns None.
    """
    return reduce(
        lambda d, key: d.get(key) if isinstance(d, dict) else None, keys, directive
    )
