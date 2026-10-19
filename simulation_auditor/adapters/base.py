"""Version independent adapter interface."""
from abc import ABC, abstractmethod
from pathlib import Path
from ..core.schema import DatasetView

class Adapter(ABC):
    version: str
    @abstractmethod
    def load(self, path: str | Path) -> DatasetView:
        """Load a validated normalized dataset."""
