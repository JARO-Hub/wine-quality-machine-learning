import json
from typing import cast

import numpy as np
import pytest
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from wine_quality.cases.case_02_model_selection.experiment_config import ExperimentConfig
from wine_quality.shared.adapters.sklearn_evaluation_engine import SklearnEvaluationEngine
from wine_quality.shared.adapters.sklearn_model_catalog import model_catalog
from wine_quality.shared.adapters.sklearn_split import build_split
from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.domain.metrics import Metrics
from wine_quality.shared.domain.types import JsonObject


def test_metrics_match_independent_hand_calculation() -> None:
    result = Metrics.calculate(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert result.mae == pytest.approx(1 / 3)
    assert result.mse == pytest.approx(1 / 3)
    assert result.rmse == pytest.approx(np.sqrt(1 / 3))
    assert result.r2 == pytest.approx(0.5)
    assert Metrics.calculate(np.ones(3), np.ones(3)).r2 is None


def test_holdout_and_folds_are_disjoint_and_reproducible(synthetic_dataset: Dataset) -> None:
    plan = build_split(synthetic_dataset, ExperimentConfig())
    repeated = build_split(synthetic_dataset, ExperimentConfig())
    np.testing.assert_array_equal(plan.train_indices, repeated.train_indices)
    assert set(plan.train_indices).isdisjoint(plan.test_indices)
    assert set(plan.train_indices) | set(plan.test_indices) == set(range(100))
    validation_rows: list[int] = []
    for fit, validation in plan.cv_folds:
        assert set(fit).isdisjoint(validation)
        assert set(fit) | set(validation) == set(range(80))
        validation_rows.extend(int(row) for row in validation)
    assert sorted(validation_rows) == list(range(80))


def test_scaler_learns_only_each_fold_training_rows(synthetic_dataset: Dataset) -> None:
    plan = build_split(synthetic_dataset, ExperimentConfig())
    features = synthetic_dataset.features[plan.train_indices]
    target = synthetic_dataset.target[plan.train_indices]
    outcome = cross_validate(
        model_catalog(42)[1].build(),
        features,
        target,
        cv=list(plan.cv_folds),
        return_estimator=True,
    )
    for estimator, (fit, _) in zip(outcome["estimator"], plan.cv_folds, strict=True):
        scaler = estimator.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, features[fit].mean(axis=0))
        np.testing.assert_allclose(scaler.var_, features[fit].var(axis=0, ddof=0))
        assert scaler.n_samples_seen_ == len(fit)


def test_ridge_matches_stated_sum_squared_error_objective(synthetic_dataset: Dataset) -> None:
    pipeline = model_catalog(42)[1].build()
    pipeline.fit(synthetic_dataset.features, synthetic_dataset.target)
    estimator = cast(Pipeline, pipeline)
    scaler = estimator.named_steps["scaler"]
    ridge = estimator.named_steps["model"]
    standardized = scaler.transform(synthetic_dataset.features)
    residuals = standardized @ ridge.coef_ + ridge.intercept_ - synthetic_dataset.target
    coefficient_gradient = 2 * standardized.T @ residuals + 2 * ridge.alpha * ridge.coef_
    np.testing.assert_allclose(coefficient_gradient, 0, atol=1e-9)
    assert float(residuals.sum()) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("model_index", [4, 5])
def test_svr_prediction_matches_support_vector_expansion(
    model_index: int, synthetic_dataset: Dataset
) -> None:
    pipeline = model_catalog(42)[model_index].build()
    pipeline.fit(synthetic_dataset.features, synthetic_dataset.target)
    steps = cast(Pipeline, pipeline).named_steps
    svr = steps["model"]
    queries = synthetic_dataset.features[:4]
    standardized = steps["scaler"].transform(queries)
    if svr.kernel == "linear":
        kernel = standardized @ svr.support_vectors_.T
    else:
        squared_distances = np.sum(
            (standardized[:, None, :] - svr.support_vectors_[None, :, :]) ** 2, axis=2
        )
        training_scaled = steps["scaler"].transform(synthetic_dataset.features)
        gamma = 1 / (training_scaled.shape[1] * training_scaled.var())
        kernel = np.exp(-gamma * squared_distances)
    reconstructed = kernel @ svr.dual_coef_.ravel() + svr.intercept_[0]
    np.testing.assert_allclose(pipeline.predict(queries), reconstructed, atol=1e-10)


def test_tree_leaves_use_mean_and_forest_averages_trees(synthetic_dataset: Dataset) -> None:
    tree_pipeline = model_catalog(42)[2].build()
    tree_pipeline.fit(synthetic_dataset.features, synthetic_dataset.target)
    tree = cast(Pipeline, tree_pipeline).named_steps["model"]
    leaves = tree.apply(synthetic_dataset.features)
    predictions = tree.predict(synthetic_dataset.features)
    for leaf in np.unique(leaves):
        np.testing.assert_allclose(
            predictions[leaves == leaf], synthetic_dataset.target[leaves == leaf].mean()
        )
    forest_pipeline = model_catalog(42)[3].build()
    forest_pipeline.fit(synthetic_dataset.features, synthetic_dataset.target)
    forest = cast(Pipeline, forest_pipeline).named_steps["model"]
    queries = synthetic_dataset.features[:4]
    expected = np.mean([estimator.predict(queries) for estimator in forest.estimators_], axis=0)
    np.testing.assert_allclose(forest_pipeline.predict(queries), expected)


def test_selection_is_cv_only_and_test_metrics_are_traceable(synthetic_dataset: Dataset) -> None:
    report = SklearnEvaluationEngine().evaluate(
        synthetic_dataset, ExperimentConfig(), model_catalog(42)[:2]
    ).report
    json.dumps(report, allow_nan=False)
    models = cast(list[JsonObject], report["models"])
    expected = min(
        models, key=lambda model: cast(float, cast(JsonObject, model["cv"])["rmse_mean"])
    )
    assert report["winner_id"] == expected["id"]
    test = cast(JsonObject, report["test"])
    assert set(test) == {"winner", "baseline"}
    assert all("test" not in model for model in models)
    split = cast(JsonObject, report["split"])
    train_ids = cast(list[int], split["train_row_ids"])
    train_mean = float(synthetic_dataset.target[train_ids].mean())
    baseline = cast(JsonObject, test["baseline"])
    baseline_predictions = cast(list[JsonObject], baseline["predictions"])
    assert all(prediction["predicted"] == train_mean for prediction in baseline_predictions)
    for result in test.values():
        outcome = cast(JsonObject, result)
        predictions = cast(list[JsonObject], outcome["predictions"])
        actual = np.array([row["actual"] for row in predictions], dtype=np.float64)
        predicted = np.array([row["predicted"] for row in predictions], dtype=np.float64)
        assert Metrics.calculate(actual, predicted).as_json() == outcome["metrics"]
