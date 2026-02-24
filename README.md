# Germinal - a pure python, lightweight LLM agent



## Installing

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
uv sync
```

This creates a virtual environment and installs all runtime dependencies.



## Running

Copy and edit the config file, then run:

```sh
cp orchestrator/config.yaml config.yaml
# edit config.yaml to set your model, paths, etc.
uv run germ
```

The orchestrator reads `config.yaml` from the current working directory by default.



## Building and Testing

Install with test dependencies:

```sh
uv sync --extra dev
```

Run the test suite:

```sh
uv run pytest tests/
```

Build a distributable wheel:

```sh
uv build
# output: dist/germinal-0.1.0-py3-none-any.whl
```



## Security

Germinal includes extensible, security validation for tool outputs. The security layer validates all tool results before they are sent to the LLM, using basic sensitive data masking and prompt injection detection.
