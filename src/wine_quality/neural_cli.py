import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from wine_quality.cases.case_01_preprocessing.service import PreprocessingService
from wine_quality.cases.case_02_model_selection.experiment_config import ExperimentConfig
from wine_quality.cases.case_03_neural_network.lesson import learning_step
from wine_quality.cases.case_03_neural_network.service import NeuralNetworkService
from wine_quality.cli import _print_selection, _save_json
from wine_quality.shared.adapters.neural_model_io import load_network, save_network
from wine_quality.shared.adapters.polars_dataset_repository import PolarsDatasetRepository
from wine_quality.shared.adapters.sklearn_evaluation_engine import SklearnEvaluationEngine
from wine_quality.shared.adapters.sklearn_neural_network import (
    extract_network,
    learning_summary,
    neural_specification,
)
from wine_quality.shared.adapters.sklearn_split import build_split
from wine_quality.shared.domain.schema import FEATURE_NAMES
from wine_quality.shared.domain.types import JsonObject
from wine_quality.shared.domain.wine_input import (
    FEATURE_LABELS,
    sample_array,
    validate_measurement,
    validate_quality,
)


def _train(data: Path, output: Path) -> None:
    config = ExperimentConfig()
    dataset = PreprocessingService(PolarsDatasetRepository()).run(data)
    print(f"Datos: {dataset.rows_clean} vinos, 11 entradas. RNA 11-8-1; cinco folds.", flush=True)
    result = NeuralNetworkService(SklearnEvaluationEngine(), neural_specification(config.seed)).run(
        dataset, config
    )
    report = result.report
    experiment = cast(JsonObject, report["experiment"])
    experiment.pop("forest_n_estimators", None)
    experiment.pop("svr_cache_mb", None)
    experiment["preprocessing"] = "exact_dedup; no_capping; StandardScaler_in_each_fold"
    experiment["scope"] = "RNA fija; prueba histórica ya observada; comparación exploratoria"
    report["neural_network"] = learning_summary(result.estimator)
    split = build_split(dataset, config)
    model = extract_network(result.estimator, dataset.features[split.train_indices])
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "report.json", report)
    save_network(output / "model.json", model, experiment)
    _print_selection(report)
    models = cast(list[JsonObject], report["models"])
    warnings = cast(list[JsonObject], models[0]["fit_warnings"])
    if warnings:
        print(f"Avisos de entrenamiento: {len(warnings)}; detalle en report.json.")
        print("No se confirmó convergencia dentro del límite fijado.")
    print(f"Modelo y resultados: {output.resolve()}")


def _read_measurements() -> list[float]:
    values: list[float] = []
    print("Introduce mediciones en sus unidades originales. Puedes usar punto o coma decimal.")
    for index, label in enumerate(FEATURE_LABELS):
        while True:
            try:
                value = float(input(f"{label}: ").strip().replace(",", "."))
                values.append(validate_measurement(index, value))
                break
            except ValueError as error:
                print(f"Entrada inválida: {error}")
    return values


def _read_quality() -> float | None:
    while True:
        raw = input("Calidad real, si la conoces (0 a 10); Enter para omitir: ").strip()
        if not raw:
            return None
        try:
            return validate_quality(float(raw.replace(",", ".")))
        except ValueError as error:
            print(f"Entrada inválida: {error}")


def _predict(
    model_path: Path,
    values: list[float] | None,
    quality: float | None,
    output: Path | None,
    trace: bool,
) -> None:
    model = load_network(model_path)
    interactive = values is None
    features = sample_array(_read_measurements() if values is None else values)
    if quality is None and interactive:
        quality = _read_quality()
    if quality is not None:
        quality = validate_quality(quality)
    standardized, hidden, predicted = model.forward(features)
    prediction = float(predicted[0])
    outside = (features[0] < model.training_min) | (features[0] > model.training_max)
    warnings = [FEATURE_NAMES[i] for i in np.flatnonzero(outside)]
    payload: JsonObject = {
        "inputs": dict(zip(FEATURE_NAMES, features[0].tolist(), strict=True)),
        "prediction": prediction,
        "known_quality": quality,
        "residual": None if quality is None else quality - prediction,
        "absolute_error": None if quality is None else abs(quality - prediction),
        "squared_error": None if quality is None else (quality - prediction) ** 2,
        "outside_training_range": list(warnings),
        "evaluation_scope": "Una entrada manual no constituye una validación independiente.",
    }
    print(f"Calidad estimada: {prediction:.4f} puntos.")
    if warnings:
        print("Fuera del intervalo de entrenamiento: " + ", ".join(warnings))
    if not 0 <= prediction <= 10:
        print("La salida lineal quedó fuera de 0 a 10. No se recortó; revisar esta predicción.")
    if quality is None:
        print("Sin calidad real: se informa predicción, no error ni precisión.")
    else:
        print(f"Real: {quality:g}; error absoluto: {abs(quality - prediction):.4f}.")
        print("Es una comprobación individual; no permite calcular R² ni generalización.")
    if trace:
        payload["standardized"] = standardized[0].tolist()
        payload["hidden_activations"] = hidden[0].tolist()
        print("Entradas estandarizadas:", np.round(standardized[0], 4))
        print("Ocho activaciones tanh:", np.round(hidden[0], 4))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        _save_json(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="RNA de vinos: entrenar, predecir y estudiar.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Validar y entrenar la RNA fija.")
    train.add_argument("--data", type=Path, default=Path("data/raw/winequality-red.csv"))
    train.add_argument("--output", type=Path, default=Path("outputs/03_neural_network"))
    predict = commands.add_parser("predict", help="Introducir un vino con el modelo ya ajustado.")
    predict.add_argument("--model", type=Path, default=Path("outputs/03_neural_network/model.json"))
    predict.add_argument("--values", type=float, nargs=11)
    predict.add_argument("--quality", type=float)
    predict.add_argument("--output", type=Path)
    predict.add_argument("--trace", action="store_true")
    commands.add_parser("lesson", help="Mostrar un paso de backpropagation con números.")
    arguments = parser.parse_args()
    try:
        if arguments.command == "train":
            _train(arguments.data, arguments.output)
        elif arguments.command == "predict":
            _predict(
                arguments.model,
                arguments.values,
                arguments.quality,
                arguments.output,
                arguments.trace,
            )
        else:
            print(json.dumps(learning_step(), indent=2, ensure_ascii=False))
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        parser.exit(2, f"Error: {error}\n")
    except (EOFError, KeyboardInterrupt):
        parser.exit(1, "\nIngreso cancelado. No se modificó el modelo.\n")


if __name__ == "__main__":
    main()
