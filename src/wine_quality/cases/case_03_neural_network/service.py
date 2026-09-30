from wine_quality.cases.case_02_model_selection.evaluation_engine import EvaluationEngine
from wine_quality.cases.case_02_model_selection.experiment_config import ExperimentConfig
from wine_quality.cases.case_02_model_selection.model_spec import ModelSpec
from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.domain.evaluation_result import EvaluationResult


class NeuralNetworkService:
    def __init__(self, engine: EvaluationEngine, specification: ModelSpec) -> None:
        self._engine = engine
        self._specification = specification

    def run(self, dataset: Dataset, config: ExperimentConfig) -> EvaluationResult:
        return self._engine.evaluate(dataset, config, (self._specification,))
