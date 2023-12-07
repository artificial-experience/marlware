FROM --platform=linux/amd64 nvidia/cuda:12.3.0-runtime-ubuntu22.04
LABEL maintainer="James Chojnacki"

# Set non-interactive frontend (to avoid interactive dialogues during apt-get install)
ENV DEBIAN_FRONTEND=noninteractive

# Set default timezone (replace 'Etc/UTC' with your desired timezone)
ENV TZ=Etc/UTC

# Ubuntu Packages
RUN apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils vim man build-essential wget sudo && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils && \
    apt-get install -y swig gcc git curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Ubuntu Packages
RUN apt-get update -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler

# Update alternatives to use Python 3.9 as the default version of Python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install Poetry
RUN curl -sSL -o install-poetry.py https://install.python-poetry.org && \
    python3 install-poetry.py --yes && \
    rm install-poetry.py

# Add Poetry to PATH
ENV PATH="$PATH:/root/.local/bin"

# Verify installation by checking the Poetry binary
RUN ls /root/.local/bin/poetry

# Copy your project files into the container at /marl
COPY . /marl

# Set the working directory to /marl
WORKDIR /marl

# Use bash to source the script, ensuring that bash is installed and used for the command
SHELL ["/bin/bash", "-c"]

# Change mode for files
RUN chmod +x activate_env.sh install_sc2.sh

# Install StarCraft II
RUN ./install_sc2.sh

# Activate
RUN ./activate_env.sh

# Set StarCraftII path
ENV SC2PATH /marl/games/StarCraftII
