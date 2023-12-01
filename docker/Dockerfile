FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

LABEL maintainer="James Chojnacki"

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Update package lists and install curl
RUN apt-get update && \
    apt-get install -y swig && \
    apt-get install -y gcc && \
    apt-get install -y git && \
    apt-get install -y python3-dev && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

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

# Install project, ensuring that the script is executable
# and using '.' which is equivalent to 'source' in bash.
RUN chmod +x activate_env.sh && . ./activate_env.sh
