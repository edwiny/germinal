import asyncio
from importlib.resources import files
from pathlib import Path

from .main_loop import main as _async_main

_CONFIG_PATH = Path.home() / ".config" / "germinal" / "config.yaml"


def _init_config() -> None:
    """
    Copy the bundled default config to ~/.config/germinal/config.yaml if it
    does not already exist.

    Uses importlib.resources so this works whether germinal is installed from
    a wheel or run directly from source via 'uv run germ'.
    """
    if _CONFIG_PATH.exists():
        return
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    default = files("orchestrator").joinpath("config.yaml").read_bytes()
    _CONFIG_PATH.write_bytes(default)
    print(f"Created default config: {_CONFIG_PATH}")
    print("Edit it before running germ again.")


def main() -> None:
    _init_config()
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
