from pathlib import Path
from typing import Protocol

from wine_quality.shared.domain.dataset import Dataset


class DatasetRepository(Protocol):
    def load(self, path: Path) -> Dataset: ...
