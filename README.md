![supported platforms](https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20%7C%20Windows%20(soon)-929292)
![supported python versions](https://img.shields.io/badge/python-%3E%3D%203.6-306998)
![license MIT](https://img.shields.io/badge/licence-MIT-green)

![MARLware Logo](docs/img/marlware-logo.png)

# MARLware

MARLware is a comprehensive framework for Multi-Agent Reinforcement Learning (MARL) based on [pymarl2](https://github.com/hijkzzz/pymarl2), designed to seamlessly integrate with the Ray engine and Hydra for efficient and scalable distributed task management. This robust platform aims to facilitate the implementation and experimentation with a variety of MARL algorithms.

## Features

* [x] **Ray Engine Integration**: Enhanced with [Ray](https://www.ray.io) for distributed task management, ensuring scalability and efficiency in complex MARL scenarios.
* [x] **Hydra Configuration**: Utilizes Hydra for dynamic and flexible configuration, streamlining the adaptation and tuning of MARL algorithms.
* [x] **Modular Design**: Built with a focus on modularity, allowing for easy integration and experimentation with different MARL algorithms.
* [x] **Python Compatibility**: Supports Python versions >= 3.6, making it accessible to a broad range of developers and researchers.

## Installation

Set up MARLware by following the installation instructions for Ray, Hydra, and other necessary dependencies:

### Environment Setup

> [!IMPORTANT]
> It is necessary to install pysc2 before using this repository, please refer to [pysc2 installation](https://github.com/google-deepmind/pysc2)

#### Poetry Environment
> source activate_env.sh

#### Docker Installation (Coming Soon)
> source install_docker.sh

## Experimentation and Usage

Effortlessly conduct advanced experiments in MARL with MARLware.

### Running Default Configurations
> python3 src/tune.py

Or specify a custom configuration:

> python3 src/tune.py --config-name="<custom_config>.yaml"

> python3 src/tune.py trainable=qmix_large


## Applications and Use Cases

MARLware is adept at handling sophisticated coordination tasks among multiple agents. Its flexibility and scalability make it suitable for strategic games, collaborative robotics, and complex multi-agent simulations.

## Join the MARLware Community

Contribute to and collaborate on MARLware as it evolves with cutting-edge technologies in Multi-Agent Reinforcement Learning.

## Reference
Inspired by existing frameworks in the field:
> [pymarl2 - GitHub](https://github.com/hijkzzz/pymarl2)

## Citation

For referencing MARLware in academic and research work:

```bibtex
@misc{chojancki2023marl-engineering,
  title={MARLware: Modular Multi-Agent Reinforcement Learning},
  author={James Chojnacki},
  year={2023},
  publisher={GitHub},
  howpublished={\url{https://github.com/marl-engineering/marlware}}
}
