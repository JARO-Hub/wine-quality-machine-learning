from collections.abc import Callable
from dataclasses import dataclass

from wine_quality.shared.domain.types import ParameterValue
from wine_quality.shared.ports.regressor import Regressor


@dataclass(frozen=True, slots=True)
class ModelSpec:
    identifier: str
    label: str
    family: str
    build: Callable[[], Regressor]
    parameter_grid: dict[str, list[ParameterValue]]
