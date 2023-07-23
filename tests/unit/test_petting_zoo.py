import pytest
from pettingzoo.sisl import multiwalker_v9


def test_petting_zoo_import_and_env_creation():
    try:
        env = multiwalker_v9.env()
        env.reset()
    except Exception as e:
        pytest.fail(f"PettingZoo environment creation failed with error {e}")

    assert env is not None, "Environment should not be None"
