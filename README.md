# Germinal - a pure python, lightweight LLM agent

**!!!WARNING!!!** this software can cause damage to your system or introduce security risks. See Risks section below.  **!!!WARNING!!!** 

## Features

* Runs anywhere Python runs
* Low memory footprint
* Approval workflow for risky actions
* Limited auto extension of tool coverage by writing and executing its own scripts

## Example usage:

```
$ git diff main | germ "generate commit messages for this diff"

$ cat application.log | germ "what might be causing the exceptions in this log file?"

# auto scripting
$ germ
Germinal interactive mode. Type your prompt and press Enter. Ctrl-D to exit.
 > what are the duplicate files in this directory?
============================================================
[APPROVAL REQUIRED]
Agent: task_agent  |  Project: default  |  Risk: high
Tool: shell_run
Parameters:
{
  "command": [
    "python",
    "~\\.local\\germinal\\find_duplicates.py"
  ]
}
============================================================
Approve? [y/N]: y
2026-02-27 09:33:59.109 WARNING [security] Sensitive data masked in tool output
Most duplicates are in the virtual environment (.venv) subdirectories, which is expected for Python projects as dependencies and vendorized files are duplicated across packages. For the actual project source code, no unintended source code duplication exists in the project itself.


$ germ --daemon
# access it via a llm client on port 8080
```

## Quickstart

At the moment, Germinal can only be installed from source. Once you've cloned the repo and changed into it, run:

```sh

# using uv (install from https://docs.astral.sh/uv/getting-started/installation/)
uv sync

# using pip
pip install pip
python -m build
pip install -e .

```

This creates a virtual environment and installs all runtime dependencies. 

Setup your authentication token on 

```sh
export OPENROUTER_API_KEY=<TOKEN>
# or powershell
$env:OPENROUTER_API_KEY="<TOKEN>"

germ
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