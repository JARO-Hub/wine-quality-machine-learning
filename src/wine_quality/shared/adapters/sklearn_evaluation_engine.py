from importlib.metadata import version
from platform import python_version
from time import perf_counter
from typing import cast
from warnings import catch_warnings, simplefilter

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from wine_quality.cases.case_02_model_selection.experiment_config import ExperimentConfig
from wine_quality.cases.case_02_model_selection.model_spec import ModelSpec
from wine_quality.shared.adapters.sklearn_split import build_split
from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.domain.evaluation_result import EvaluationResult
from wine_quality.shared.domain.fit_result import FitResult
from wine_quality.shared.domain.metrics import Metrics
from wine_quality.shared.domain.split_plan import SplitPlan
from wine_quality.shared.domain.types import FloatArray, IndexArray, JsonObject, JsonValue
from wine_quality.shared.ports.regressor import Regressor


def dataset_summary(dataset: Dataset) -> JsonObject:
    qualities, counts = np.unique(dataset.target, return_counts=True)
    return {
        "rows_raw": dataset.rows_raw,
        "rows_clean": dataset.rows_clean,
        "duplicates_removed": dataset.duplicates_removed,
        "quality_counts": {str(int(q)): int(n) for q, n in zip(qualities, counts, strict=True)},
        "matrix_bytes": {
            "features": int(dataset.features.nbytes),
            "target": int(dataset.target.nbytes),
        },
    }


def _indices_as_json(indices: IndexArray) -> list[JsonValue]:
    return [int(index) for index in indices]


def _split_as_json(dataset: Dataset, split: SplitPlan) -> JsonObject:
    train_ids = dataset.row_ids[split.train_indices]
    return {
        "train_row_ids": _indices_as_json(train_ids),
        "test_row_ids": _indices_as_json(dataset.row_ids[split.test_indices]),
        "train_count": int(split.train_indices.size),
        "test_count": int(split.test_indices.size),
        "cv_folds": [
            {
                "train_row_ids": _indices_as_json(train_ids[fit]),
                "validation_row_ids": _indices_as_json(train_ids[validation]),
                "train_count": int(fit.size),
                "validation_count": int(validation.size),
            }
            for fit, validation in split.cv_folds
        ],
    }


def tune_model(
    specification: ModelSpec, features: FloatArray, target: FloatArray, split: SplitPlan
) -> FitResult:
    started = perf_counter()
    search = GridSearchCV(
        estimator=specification.build(),
        param_grid=specification.parameter_grid,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        },
        refit="rmse",
        cv=list(split.cv_folds),
        n_jobs=1,
        error_score="raise",
    )
    with catch_warnings(record=True) as captured_warnings:
        simplefilter("always", ConvergenceWarning)
        search.fit(features, target)
    elapsed = perf_counter() - started
    estimator = cast(Regressor, search.best_estimator_)
    best_index = int(search.best_index_)
    results = search.cv_results_
    cv: JsonObject = {
        "rmse_mean": float(-results["mean_test_rmse"][best_index]),
        "rmse_std": float(results["std_test_rmse"][best_index]),
        "mae_mean": float(-results["mean_test_mae"][best_index]),
        "mse_mean": float(-results["mean_test_mse"][best_index]),
        "r2_mean": float(results["mean_test_r2"][best_index]),
        "fold_rmse": [
            float(-results[f"split{fold}_test_rmse"][best_index])
            for fold in range(len(split.cv_folds))
        ],
    }
    summary: JsonObject = {
        "id": specification.identifier,
        "label": specification.label,
        "family": specification.family,
        "best_params": cast(JsonObject, dict(search.best_params_)),
        "parameter_grid": {
            key: list(values) for key, values in specification.parameter_grid.items()
        },
        "candidate_count": int(len(results["params"])),
        "cv": cv,
        "train": Metrics.calculate(target, estimator.predict(features)).as_json(),
        "fit_seconds": elapsed,
        "fit_warnings": [
            {"category": warning.category.__name__, "message": str(warning.message)}
            for warning in captured_warnings
        ],
    }
    return FitResult(specification.identifier, estimator, cast(float, cv["rmse_mean"]), summary)


def _test_summary(
    fit: FitResult, features: FloatArray, target: FloatArray, row_ids: IndexArray
) -> JsonObject:
    predicted = fit.estimator.predict(features)
    return {
        "id": fit.identifier,
        "metrics": Metrics.calculate(target, predicted).as_json(),
        "predictions": [
            {"row_id": int(row_id), "actual": float(actual), "predicted": float(prediction)}
            for row_id, actual, prediction in zip(row_ids, target, predicted, strict=True)
        ],
    }


class SklearnEvaluationEngine:
    def evaluate(
        self, dataset: Dataset, config: ExperimentConfig, models: tuple[ModelSpec, ...]
    ) -> EvaluationResult:
        if not models or len({model.identifier for model in models}) != len(models):
            raise ValueError("Se requiere un catálogo no vacío con identificadores únicos.")
        started = perf_counter()
        split = build_split(dataset, config)
        train_features = dataset.features[split.train_indices]
        train_target = dataset.target[split.train_indices]
        fitted = [tune_model(model, train_features, train_target, split) for model in models]
        winner = min(fitted, key=lambda result: result.cv_rmse)
        baseline = tune_model(
            ModelSpec(
                "dummy_mean",
                "Referencia: media",
                "baseline",
                lambda: cast(Regressor, DummyRegressor(strategy="mean")),
                {},
            ),
            train_features,
            train_target,
            split,
        )
        test_features = dataset.features[split.test_indices]
        test_target = dataset.target[split.test_indices]
        test_ids = dataset.row_ids[split.test_indices]
        report: JsonObject = {
            "schema_version": "1.0",
            "experiment": {
                "target": "quality",
                "task": "regression_on_ordinal_score",
                "primary_metric": "rmse",
                "test_fraction": config.test_fraction,
                "seed": config.seed,
                "cv_folds": config.cv_folds,
                "cv_shuffle": True,
                "cv_strategy": "KFold",
                "holdout_strategy": "stratified_by_quality",
                "selection_rule": "lowest_mean_cv_rmse; exact_ties_follow_catalog_order",
                "preprocessing": (
                    "exact_dedup; no_capping; StandardScaler_in_linear_and_SVR_pipelines"
                ),
                "feature_names": list(dataset.feature_names),
                "input_sha256": dataset.input_sha256,
                "versions": {
                    "python": python_version(),
                    "numpy": version("numpy"),
                    "polars": version("polars"),
                    "scikit-learn": version("scikit-learn"),
                },
                "n_jobs": 1,
                "forest_n_estimators": 160,
                "svr_cache_mb": 64,
                "row_id_definition": "zero_based_original_data_row_before_deduplication",
            },
            "data": dataset_summary(dataset),
            "split": _split_as_json(dataset, split),
            "models": [fit.summary for fit in fitted],
            "winner_id": winner.identifier,
            "baseline_cv": baseline.summary,
            "test": {
                "winner": _test_summary(winner, test_features, test_target, test_ids),
                "baseline": _test_summary(baseline, test_features, test_target, test_ids),
            },
            "total_seconds": perf_counter() - started,
        }
        return EvaluationResult(report, winner.estimator)
