#/bin/sh
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"
poetry --version
poetry install
