from dataclasses import dataclass

import numpy as np

from wine_quality.shared.domain.types import FloatArray
from wine_quality.shared.domain.wine_input import validate_features


@dataclass(frozen=True, slots=True)
class NeuralNetwork:
    mean: FloatArray
    scale: FloatArray
    hidden_weights: FloatArray
    hidden_bias: FloatArray
    output_weights: FloatArray
    output_bias: FloatArray
    training_min: FloatArray
    training_max: FloatArray

    def __post_init__(self) -> None:
        expected = (
            (self.mean, (11,)),
            (self.scale, (11,)),
            (self.hidden_weights, (11, 8)),
            (self.hidden_bias, (8,)),
            (self.output_weights, (8, 1)),
            (self.output_bias, (1,)),
            (self.training_min, (11,)),
            (self.training_max, (11,)),
        )
        for values, shape in expected:
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(
                    f"Parámetros inválidos: se esperaba forma {shape} y números finitos."
                )
            values.setflags(write=False)
        if np.any(self.scale <= 0) or np.any(self.training_min > self.training_max):
            raise ValueError("Escalas o límites de entrenamiento inválidos.")

    def forward(self, features: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        validate_features(features)
        standardized = (features - self.mean) / self.scale
        hidden = np.tanh(standardized @ self.hidden_weights + self.hidden_bias)
        predictions = (hidden @ self.output_weights + self.output_bias).ravel()
        return standardized, hidden, predictions

    def predict(self, features: FloatArray) -> FloatArray:
        return self.forward(features)[2]
