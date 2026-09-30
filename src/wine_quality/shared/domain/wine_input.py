import math

import numpy as np

from wine_quality.shared.domain.schema import FEATURE_NAMES
from wine_quality.shared.domain.types import FloatArray

FEATURE_LABELS: tuple[str, ...] = (
    "Acidez fija (g/dm³)",
    "Acidez volátil (g/dm³)",
    "Ácido cítrico (g/dm³)",
    "Azúcar residual (g/dm³)",
    "Cloruros (g/dm³)",
    "Dióxido de azufre libre (mg/dm³)",
    "Dióxido de azufre total (mg/dm³)",
    "Densidad (g/cm³)",
    "pH",
    "Sulfatos (g/dm³)",
    "Alcohol (% vol.)",
)


def validate_measurement(index: int, value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{FEATURE_NAMES[index]} debe ser finito y no negativo.")
    if index == 7 and value == 0:
        raise ValueError("La densidad debe ser mayor que cero.")
    if index == 8 and value > 14:
        raise ValueError("El pH debe estar entre 0 y 14.")
    if index == 10 and value > 100:
        raise ValueError("El alcohol debe estar entre 0 y 100 %.")
    return value


def validate_quality(value: float) -> float:
    if not math.isfinite(value) or not value.is_integer() or not 0 <= value <= 10:
        raise ValueError("La calidad real debe ser un entero entre 0 y 10.")
    return value


def validate_features(features: FloatArray) -> None:
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES) or not features.shape[0]:
        raise ValueError("Se necesitan una o más filas de once características.")
    for row in features:
        for index, value in enumerate(row):
            validate_measurement(index, float(value))


def sample_array(values: list[float]) -> FloatArray:
    features = np.asarray([values], dtype=np.float64)
    validate_features(features)
    return features
