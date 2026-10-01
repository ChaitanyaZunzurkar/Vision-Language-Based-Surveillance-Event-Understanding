"""Path management utilities for surveillance backend."""

from pathlib import Path
from typing import Optional


class ProjectPaths:
    """Manages project directories and ensures required folders exist."""

    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is None:
            # Assume project root is two levels up from src/utils/
            self.root_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self.root_dir = Path(root_dir).resolve()

        self.configs_dir = self.root_dir / "configs"
        self.config_yaml = self.configs_dir / "config.yaml"
        self.classes_yaml = self.configs_dir / "classes.yaml"

        self.data_dir = self.root_dir / "data"
        self.uploads_dir = self.data_dir / "uploads"
        self.outputs_dir = self.data_dir / "outputs"
        self.clips_dir = self.outputs_dir / "clips"
        self.tracks_dir = self.outputs_dir / "tracks"
        self.windows_dir = self.outputs_dir / "windows"
        self.vectors_dir = self.data_dir / "vectors"

        self.weights_dir = self.root_dir / "weights"
        self.db_path = self.root_dir / "events.db"

        self.ensure_directories()

    def ensure_directories(self) -> None:
        """Create required directories if they don't already exist."""
        for directory in [
            self.configs_dir,
            self.data_dir,
            self.uploads_dir,
            self.outputs_dir,
            self.clips_dir,
            self.tracks_dir,
            self.windows_dir,
            self.vectors_dir,
            self.weights_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Default global instance
paths = ProjectPaths()
