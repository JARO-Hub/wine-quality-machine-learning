from dataclasses import dataclass

import numpy as np

from wine_quality.shared.domain.types import FloatArray, JsonObject


@dataclass(frozen=True, slots=True)
class Metrics:
    mae: float
    mse: float
    rmse: float
    r2: float | None

    @classmethod
    def calculate(cls, actual: FloatArray, predicted: FloatArray) -> "Metrics":
        if actual.ndim != 1 or actual.shape != predicted.shape or actual.size < 2:
            raise ValueError(
                "Las métricas requieren vectores iguales y al menos dos observaciones."
            )
        if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
            raise ValueError("Las métricas requieren valores finitos.")
        residuals = actual - predicted
        mse = float(np.mean(np.square(residuals)))
        total_variation = float(np.sum(np.square(actual - actual.mean())))
        r2 = None if total_variation == 0 else 1 - float(np.sum(residuals**2)) / total_variation
        return cls(float(np.mean(np.abs(residuals))), mse, float(np.sqrt(mse)), r2)

    def as_json(self) -> JsonObject:
        return {"mae": self.mae, "mse": self.mse, "rmse": self.rmse, "r2": self.r2}
