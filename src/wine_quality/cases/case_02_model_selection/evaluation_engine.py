from typing import Protocol

from wine_quality.cases.case_02_model_selection.experiment_config import ExperimentConfig
from wine_quality.cases.case_02_model_selection.model_spec import ModelSpec
from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.domain.types import JsonObject


class EvaluationEngine(Protocol):
    def evaluate(
        self, dataset: Dataset, config: ExperimentConfig, models: tuple[ModelSpec, ...]
    ) -> JsonObject: ...
