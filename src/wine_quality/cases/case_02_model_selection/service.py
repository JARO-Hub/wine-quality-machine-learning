from wine_quality.cases.case_02_model_selection.evaluation_engine import EvaluationEngine
from wine_quality.cases.case_02_model_selection.experiment_config import ExperimentConfig
from wine_quality.cases.case_02_model_selection.model_spec import ModelSpec
from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.domain.types import JsonObject


class ModelSelectionService:
    def __init__(self, engine: EvaluationEngine, models: tuple[ModelSpec, ...]) -> None:
        self._engine = engine
        self._models = models

    def run(self, dataset: Dataset, config: ExperimentConfig) -> JsonObject:
        return self._engine.evaluate(dataset, config, self._models).report
