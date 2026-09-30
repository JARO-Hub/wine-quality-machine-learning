import json
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from wine_quality.cases.case_02_model_selection.experiment_config import ExperimentConfig
from wine_quality.cases.case_03_neural_network.lesson import gradient, loss
from wine_quality.shared.adapters.neural_model_io import load_network, save_network
from wine_quality.shared.adapters.sklearn_evaluation_engine import SklearnEvaluationEngine
from wine_quality.shared.adapters.sklearn_neural_network import (
    extract_network,
    neural_specification,
)
from wine_quality.shared.adapters.sklearn_split import build_split
from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.domain.types import JsonObject
from wine_quality.shared.domain.wine_input import sample_array, validate_quality


def test_backpropagation_matches_finite_differences_and_reduces_loss() -> None:
    inputs = np.array([1.0, 0.5])
    parameters = np.array([0.2, 0.4, 0.1, 0.6, 0.1])
    target, epsilon = 0.8, 1e-6
    analytical = gradient(parameters, inputs, target)
    numerical = np.empty_like(parameters)
    for index in range(parameters.size):
        direction = np.zeros_like(parameters)
        direction[index] = epsilon
        numerical[index] = (
            loss(parameters + direction, inputs, target)
            - loss(parameters - direction, inputs, target)
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-9)
    assert loss(parameters - 0.1 * analytical, inputs, target) < loss(parameters, inputs, target)


def test_saved_network_matches_pipeline_without_learning(
    synthetic_frame: pl.DataFrame, tmp_path: Path
) -> None:
    features = synthetic_frame.drop("quality").to_numpy()
    target = synthetic_frame["quality"].to_numpy()
    pipeline = cast(Pipeline, neural_specification(42).build())
    pipeline.set_params(model__max_iter=2, model__batch_size=32)
    with pytest.warns(ConvergenceWarning):
        pipeline.fit(features[:40], target[:40])
    network = extract_network(pipeline, features[:40])
    path = tmp_path / "model.json"
    save_network(path, network, {"seed": 42})
    before = path.read_bytes()
    loaded = load_network(path)
    np.testing.assert_allclose(loaded.predict(features[40:]), pipeline.predict(features[40:]))
    np.testing.assert_allclose(loaded.mean, features[:40].mean(axis=0))
    np.testing.assert_allclose(loaded.training_max, features[:40].max(axis=0))
    assert path.read_bytes() == before
    assert not loaded.hidden_weights.flags.writeable
    raw = json.loads(before)
    raw["feature_names"][0] = "quality"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="columnas"):
        load_network(path)
    raw["feature_names"][0] = "fixed acidity"
    raw["arrays"]["scale"][0] = 0
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Escalas"):
        load_network(path)


def test_neural_scaler_is_fit_within_each_fold(synthetic_dataset: Dataset) -> None:
    plan = build_split(synthetic_dataset, ExperimentConfig())
    features = synthetic_dataset.features[plan.train_indices]
    target = synthetic_dataset.target[plan.train_indices]
    pipeline = cast(Pipeline, neural_specification(42).build())
    pipeline.set_params(model__max_iter=2, model__batch_size=32)
    with pytest.warns(ConvergenceWarning):
        outcome = cross_validate(
            pipeline, features, target, cv=list(plan.cv_folds), return_estimator=True
        )
    for estimator, (fit, _) in zip(outcome["estimator"], plan.cv_folds, strict=True):
        scaler = estimator.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, features[fit].mean(axis=0))
        assert scaler.n_samples_seen_ == len(fit)


def test_convergence_warnings_are_persisted(synthetic_dataset: Dataset) -> None:
    from wine_quality.cases.case_02_model_selection.model_spec import ModelSpec

    pipeline = cast(Pipeline, neural_specification(42).build())
    pipeline.set_params(model__max_iter=1, model__batch_size=32)
    spec = ModelSpec("short_run", "Test RNA", "neural", lambda: pipeline, {})
    report = (
        SklearnEvaluationEngine().evaluate(synthetic_dataset, ExperimentConfig(), (spec,)).report
    )
    models = cast(list[JsonObject], report["models"])
    warnings = cast(list[JsonObject], models[0]["fit_warnings"])
    assert len(warnings) == 6
    assert all(item["category"] == "ConvergenceWarning" for item in warnings)


@pytest.mark.parametrize(
    "index,value",
    [(0, -1.0), (2, float("nan")), (3, float("inf")), (7, 0.0), (8, 15.0), (10, 101.0)],
)
def test_invalid_manual_measurements_are_rejected(index: int, value: float) -> None:
    values = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    values[index] = value
    with pytest.raises(ValueError):
        sample_array(values)


@pytest.mark.parametrize("quality", [-1.0, 5.5, 11.0, float("nan")])
def test_known_quality_must_be_a_valid_score(quality: float) -> None:
    with pytest.raises(ValueError):
        validate_quality(quality)
