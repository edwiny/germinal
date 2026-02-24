# Germinal - a pure python, lightweight LLM agent

**!!!WARNING!!!** this software can cause damage to your system or introduce security risks. See Risks section below.  **!!!WARNING!!!** 

## Quickstart

At the moment, Germinal can only be installed from source. Once you've cloned the repo and changed into it, run:

```sh
uv sync
```
Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

This creates a virtual environment and installs all runtime dependencies. 

Setup your authentication token on 

```sh
export OPENROUTER_API_KEY=<TOKEN>
# or powershell
$env:OPENROUTER_API_KEY="<TOKEN>"

uv run germ
```

You can add other models in the config:

```sh
nano ${HOME}/.config/germinal/config.yaml
```



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

Germinal includes extensible security validation for tool outputs.

The security layer validates tool results before they are sent to the LLM, using basic sensitive data masking and prompt injection detection.

Tool use and folder access is configurable via allow lists in the config.


## Risks

* The agent can supplement it's tool coverage by writing and executing Python scripts, which may contain bugs or have unintended consequences.
* Prompt injection.
* Accidental or unintended mutation of the host operating environment.
* Exfiltration of your data to 3rd parties.
* Burning through your token credit balance.