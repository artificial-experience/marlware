echo "Building Dockerfile with image name marl-engineering:1.0"
docker build --no-cache -f docker/Dockerfile -t marl-engineering:1.0 .
docker run -it --rm marl-engineering:1.0
