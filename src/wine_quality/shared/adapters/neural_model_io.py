import json
from pathlib import Path
from typing import cast

import numpy as np

from wine_quality.shared.domain.neural_network import NeuralNetwork
from wine_quality.shared.domain.schema import FEATURE_NAMES
from wine_quality.shared.domain.types import JsonObject

ARRAY_NAMES = (
    "mean",
    "scale",
    "hidden_weights",
    "hidden_bias",
    "output_weights",
    "output_bias",
    "training_min",
    "training_max",
)


def save_network(path: Path, model: NeuralNetwork, provenance: JsonObject) -> None:
    payload: JsonObject = {
        "format": "wine-neural-11-8-1-v1",
        "feature_names": list(FEATURE_NAMES),
        "hidden_activation": "tanh",
        "output_activation": "identity",
        "provenance": provenance,
        "arrays": {name: getattr(model, name).tolist() for name in ARRAY_NAMES},
    }
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_network(path: Path) -> NeuralNetwork:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("El modelo debe ser un objeto JSON.")
    data = cast(dict[str, object], raw)
    if (
        data.get("format") != "wine-neural-11-8-1-v1"
        or data.get("feature_names") != list(FEATURE_NAMES)
        or data.get("hidden_activation") != "tanh"
        or data.get("output_activation") != "identity"
        or not isinstance(data.get("arrays"), dict)
    ):
        raise ValueError("Formato, columnas o activaciones incompatibles.")
    arrays = cast(dict[str, object], data["arrays"])
    try:
        return NeuralNetwork(*(np.asarray(arrays[name], dtype=np.float64) for name in ARRAY_NAMES))
    except (KeyError, TypeError) as error:
        raise ValueError("Faltan parámetros numéricos del modelo.") from error
