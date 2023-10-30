from qmix.environment import SC2Environment


def test_sc2_env_creation():
    config = {
        "prefix": {"choice": "8m"},
    }
    env_creator = SC2Environment(config)
    env, info = env_creator.create_env_instance()
    assert env is not None, "Env should not be None"
