# Contributing to LLMSmith

## First time setup

Here are the guidelines to set up your environment after you have cloned the [LLMSmith repository](https://github.com/dheerajgopi/llmsmith).

This project uses Python version 3.9.1 and [Poetry](https://python-poetry.org/) as the dependency managament and packaging tool.
Python version can be managed using [pyenv](https://github.com/pyenv/pyenv).

### Setting up the virtual environment

- Install python 3.9.1 using `pyenv` with the following command.

  `pyenv install 3.9.1`

- `cd` into the LLMSmith repository root directory.

- Use the below `pyenv` command to override Python version at a directory level, so that whenever you `cd` into the LLMSmith repo, `python` command will point to version `3.9.1`.

  `pyenv local 3.9.1`

- Now, instruct Poetry to use Python 3.9.1, using the below command.

  `poetry env use $(pyenv which python)`

  You might have to set `virtualenvs.prefer-active-python` option to `true` in Poetry for the above command to work. Check [this](https://python-poetry.org/docs/managing-environments/) for more info.

- Activate the virtual environment using the below command.

  `. path/to/llmsmith/.venv/bin/activate`

- Install the dependencies using poetry.

  `poetry install --with dev,docs,test -E all`

## Linting and formatting

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

`ruff check . --fix` - Runs the linter.
`ruff format` - Runs the formatter.
