"""Configuration loader for surveillance event pipeline."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from src.utils.paths import paths
from src.utils.logger import logger


class ConfigLoader:
    """Loads and provides access to project configuration and class definitions."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        classes_path: Optional[Path] = None,
    ):
        self.config_path = config_path or paths.config_yaml
        self.classes_path = classes_path or paths.classes_yaml
        self._config: Dict[str, Any] = {}
        self._classes: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration files from disk."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {self.config_path}")
        else:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            self._config = {}

        if self.classes_path.exists():
            with open(self.classes_path, "r", encoding="utf-8") as f:
                self._classes = yaml.safe_load(f) or {}
            logger.info(f"Loaded class definitions from {self.classes_path}")
        else:
            logger.warning(f"Classes file not found at {self.classes_path}")
            self._classes = {}

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def classes(self) -> Dict[str, Any]:
        return self._classes

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'detection.conf_threshold')."""
        parts = key_path.split(".")
        current = self._config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    @property
    def detection_classes(self) -> list:
        return self._classes.get("detection_classes", [])

    @property
    def anomaly_categories(self) -> list:
        return self._classes.get("anomaly_categories", [])

    @property
    def event_ontology(self) -> list:
        return self._classes.get("event_ontology", [])


# Default global config loader
config_loader = ConfigLoader()
