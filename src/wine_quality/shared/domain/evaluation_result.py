from dataclasses import dataclass

from wine_quality.shared.domain.types import JsonObject
from wine_quality.shared.ports.regressor import Regressor


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    report: JsonObject
    estimator: Regressor
