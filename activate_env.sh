DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))
MARLROOT=$DIR
export MARLROOT

poetry lock
poetry install
source $(poetry env info --path)/bin/activate
pre-commit install
echo "MARL Engineering Activated"

echo "Running Tests..."
pytest ./tests -W ignore::DeprecationWarning
