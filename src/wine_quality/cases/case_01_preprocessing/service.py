from pathlib import Path

from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.ports.dataset_repository import DatasetRepository


class PreprocessingService:
    def __init__(self, repository: DatasetRepository) -> None:
        self._repository = repository

    def run(self, path: Path) -> Dataset:
        return self._repository.load(path)
