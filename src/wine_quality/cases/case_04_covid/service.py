from importlib.metadata import version
from platform import python_version
from time import perf_counter
from typing import cast

import numpy as np

from wine_quality.cases.case_02_model_selection.model_spec import ModelSpec
from wine_quality.cases.case_04_covid.config import CovidConfig
from wine_quality.cases.case_04_covid.preprocessing import series_summary
from wine_quality.shared.adapters.sklearn_evaluation_engine import tune_model
from wine_quality.shared.adapters.sklearn_temporal_split import build_temporal_split
from wine_quality.shared.domain.forecast_dataset import ForecastDataset
from wine_quality.shared.domain.metrics import Metrics
from wine_quality.shared.domain.types import FloatArray, IndexArray, JsonObject, JsonValue


def _date_block(dataset: ForecastDataset, indices: IndexArray) -> JsonObject:
    return {
        "count": int(indices.size),
        "first_date": dataset.target_dates[int(indices[0])].isoformat(),
        "last_date": dataset.target_dates[int(indices[-1])].isoformat(),
        "target_date_indices": [int(value) for value in dataset.target_date_indices[indices]],
    }


def _prediction_summary(
    dataset: ForecastDataset, indices: IndexArray, prediction: FloatArray, identifier: str
) -> JsonObject:
    return {
        "id": identifier,
        "metrics": Metrics.calculate(dataset.target[indices], prediction).as_json(),
        "negative_prediction_count": int(np.count_nonzero(prediction < 0)),
        "predictions": [
            {
                "date": dataset.target_dates[int(index)].isoformat(),
                "target_date_index": int(dataset.target_date_indices[index]),
                "actual": float(dataset.target[index]),
                "predicted": float(predicted),
            }
            for index, predicted in zip(indices, prediction, strict=True)
        ],
    }


class CovidExperimentService:
    def __init__(self, models: tuple[ModelSpec, ...]) -> None:
        self._models = models

    def run(self, dataset: ForecastDataset, config: CovidConfig) -> JsonObject:
        if not self._models or len({m.identifier for m in self._models}) != len(self._models):
            raise ValueError("Se requieren modelos con identificadores únicos.")
        started = perf_counter()
        split = build_temporal_split(dataset, config)
        features = dataset.features[split.train_indices]
        target = dataset.target[split.train_indices]
        fitted = [tune_model(model, features, target, split) for model in self._models]
        winner = min(fitted, key=lambda result: result.cv_rmse)
        baseline_cv: list[JsonObject] = []
        test: JsonObject = {
            "winner": _prediction_summary(
                dataset,
                split.test_indices,
                winner.estimator.predict(dataset.features[split.test_indices]),
                winner.identifier,
            )
        }
        for identifier, lag in (("persistence_1", 1), ("persistence_7", 7)):
            fold_metrics = [
                Metrics.calculate(target[validation], features[validation, lag - 1]).as_json()
                for _, validation in split.cv_folds
            ]
            scores = [cast(float, item["rmse"]) for item in fold_metrics]
            baseline_cv.append(
                {
                    "id": identifier,
                    "lag": lag,
                    "rmse_mean": float(np.mean(scores)),
                    "rmse_std": float(np.std(scores)),
                    "fold_metrics": cast(list[JsonValue], fold_metrics),
                }
            )
            test[identifier] = _prediction_summary(
                dataset,
                split.test_indices,
                dataset.features[split.test_indices, lag - 1],
                identifier,
            )
        summaries: list[JsonObject] = []
        for fit in fitted:
            fit.summary["estimator_parameters"] = {
                name: repr(value)
                for name, value in sorted(fit.estimator.get_params(deep=True).items())
                if "__" in name
            }
            summaries.append(fit.summary)
        return {
            "schema_version": "covid-1.0",
            "experiment": {
                "protocol": "docs/decisions/004-covid-temporal.md",
                "task": "one_day_ahead_registered_daily_change",
                "horizon_days": 1,
                "country": dataset.series.country,
                "primary_metric": "rmse",
                "seed": config.seed,
                "test_fraction": config.test_fraction,
                "cv_folds": config.cv_folds,
                "cv_strategy": "TimeSeriesSplit_expanding",
                "cv_shuffle": False,
                "gap": 0,
                "selection_rule": "lowest_mean_cv_rmse; exact_ties_follow_catalog_order",
                "test_policy": "winner_and_two_fixed_baselines_only",
                "test_mode": "fixed_parameters; observed_past_updated_daily; horizon_1",
                "test_history": (
                    "First evaluation under this COVID protocol; later reuse must be declared."
                ),
                "snapshot_scope": "Retrospective CSV; historical publication vintages unavailable.",
                "preprocessing": (
                    "geographic_exact_dedup; no_clipping; past_lags; fold_X_and_y_scaling"
                ),
                "feature_names": list(dataset.feature_names),
                "input_sha256": dataset.series.input_sha256,
                "n_jobs": 1,
                "versions": {
                    "python": python_version(),
                    "numpy": version("numpy"),
                    "polars": version("polars"),
                    "scikit-learn": version("scikit-learn"),
                },
            },
            "data": {
                **series_summary(dataset.series),
                "supervised_count": int(dataset.target.size),
                "excluded_initial_dates": 15,
                "matrix_bytes": {
                    "features": int(dataset.features.nbytes),
                    "target": int(dataset.target.nbytes),
                    "scope": "NumPy buffers only; not peak process RAM",
                },
            },
            "split": {
                "train": _date_block(dataset, split.train_indices),
                "test": _date_block(dataset, split.test_indices),
                "cv_folds": [
                    {
                        "train": _date_block(dataset, fit),
                        "validation": _date_block(dataset, validation),
                    }
                    for fit, validation in split.cv_folds
                ],
            },
            "models": cast(list[JsonValue], summaries),
            "winner_id": winner.identifier,
            "baseline_cv": cast(list[JsonValue], baseline_cv),
            "test": test,
            "total_seconds": perf_counter() - started,
        }
