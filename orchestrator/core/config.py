"""
Configuration management for the Germinal orchestrator.

This module provides a singleton Config class that loads configuration from
config.yaml and handles OS-agnostic path expansion.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


class Config:
    """
    Singleton configuration class that loads and manages application config.

    Handles path expansion for OS-agnostic path resolution, expanding ~ to
    the user's home directory and resolving relative paths to absolute paths.
    """

    _instance = None
    _config_data: Dict[str, Any] = {}

    def __new__(cls) -> 'Config':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not self._config_data:  # Only load if not already loaded
            self._load_config()

    def _load_config(self, path: str | None = None) -> None:
        """
        Load configuration from YAML file with path expansion.

        Args:
            path: Optional path to config file. If None, uses default resolution.
        """
        # Resolve config path: explicit arg > ~/.config/germinal/config.yaml > local fallback
        resolved = self._resolve_config_path(path)

        with open(resolved) as f:
            self._config_data = yaml.safe_load(f)

        # Expand paths only in the 'paths' section
        if 'paths' in self._config_data:
            self._config_data['paths'] = self._expand_paths(self._config_data['paths'])

    def _resolve_config_path(self, path: str | None) -> Path:
        """Resolve the config file path using the same logic as main_loop.py."""
        if path:
            return Path(path)

        # Default user config location
        user_config = Path.home() / ".config" / "germinal" / "config.yaml"
        if user_config.exists():
            return user_config

        # Fallback to local config.yaml for development
        local_config = Path("config.yaml")
        if local_config.exists():
            return local_config

        raise FileNotFoundError(
            f"Config file not found. Searched: {user_config}, {local_config}"
        )

    def _expand_paths(self, data: Any) -> Any:
        """
        Recursively expand paths in configuration data.

        Handles strings, lists, and nested dictionaries. Expands ~ to home
        directory and resolves relative paths to absolute paths.
        """
        if isinstance(data, dict):
            return {key: self._expand_paths(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._expand_paths(item) for item in data]
        elif isinstance(data, str):
            return self._expand_path_string(data)
        else:
            return data

    def _expand_path_string(self, path_str: str) -> str:
        """
        Expand a path string to be OS-agnostic.

        - Expands environment variables like $HOME, ${HOME}
        - Expands ~ to the user's home directory
        - Resolves relative paths to absolute paths
        - Makes all paths absolute for consistency
        """
        # First expand environment variables
        expanded_str = os.path.expandvars(path_str)

        path = Path(expanded_str)

        # Expand ~ to home directory (in case it wasn't expanded by expandvars)
        path = path.expanduser()

        # Always resolve to absolute path for consistency
        # This ensures all paths are absolute and OS-agnostic
        return str(path.resolve())

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: Dot-separated key path (e.g., "paths.db")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config_data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        try:
            self[key]
            return True
        except KeyError:
            return False

    def data(self) -> Dict[str, Any]:
        """Get the full configuration data dictionary."""
        return self._config_data.copy()

    def reload(self, path: str | None = None) -> None:
        """
        Reload configuration from file.

        Args:
            path: Optional path to config file
        """
        self._config_data.clear()
        self._load_config(path)


# Global config instance
config = Config()