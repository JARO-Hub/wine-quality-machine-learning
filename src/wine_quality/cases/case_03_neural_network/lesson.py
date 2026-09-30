import math

import numpy as np

from wine_quality.shared.domain.types import FloatArray, JsonObject

PARAMETER_NAMES = ("w1", "w2", "b", "v", "c")


def forward(parameters: FloatArray, inputs: FloatArray) -> tuple[float, float]:
    hidden = math.tanh(float(parameters[:2] @ inputs + parameters[2]))
    prediction = float(parameters[3] * hidden + parameters[4])
    return hidden, prediction


def loss(parameters: FloatArray, inputs: FloatArray, target: float) -> float:
    return 0.5 * (forward(parameters, inputs)[1] - target) ** 2


def gradient(parameters: FloatArray, inputs: FloatArray, target: float) -> FloatArray:
    hidden, prediction = forward(parameters, inputs)
    output_delta = prediction - target
    hidden_delta = output_delta * parameters[3] * (1 - hidden**2)
    return np.array(
        [
            hidden_delta * inputs[0],
            hidden_delta * inputs[1],
            hidden_delta,
            output_delta * hidden,
            output_delta,
        ],
        dtype=np.float64,
    )


def learning_step() -> JsonObject:
    inputs = np.array([1.0, 0.5])
    parameters = np.array([0.2, 0.4, 0.1, 0.6, 0.1])
    target, rate = 0.8, 0.1
    hidden, prediction = forward(parameters, inputs)
    derivatives = gradient(parameters, inputs, target)
    updated = parameters - rate * derivatives
    return {
        "purpose": "Ejercicio didáctico 2-1-1; no son pesos del vino.",
        "inputs": inputs.tolist(),
        "target": target,
        "learning_rate": rate,
        "hidden": hidden,
        "prediction_before": prediction,
        "loss_before": loss(parameters, inputs, target),
        "parameters_before": dict(zip(PARAMETER_NAMES, parameters.tolist(), strict=True)),
        "gradient": dict(zip(PARAMETER_NAMES, derivatives.tolist(), strict=True)),
        "parameters_after": dict(zip(PARAMETER_NAMES, updated.tolist(), strict=True)),
        "prediction_after": forward(updated, inputs)[1],
        "loss_after": loss(updated, inputs, target),
    }
