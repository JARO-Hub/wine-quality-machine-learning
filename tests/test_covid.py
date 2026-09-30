import json
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import pytest
from sklearn.model_selection import cross_validate

from wine_quality.cases.case_04_covid.config import CovidConfig
from wine_quality.cases.case_04_covid.preprocessing import CovidPreprocessingService, series_summary
from wine_quality.cases.case_04_covid.service import CovidExperimentService
from wine_quality.shared.adapters.polars_covid_repository import PolarsCovidRepository
from wine_quality.shared.adapters.sklearn_covid_catalog import covid_model_catalog
from wine_quality.shared.adapters.sklearn_temporal_split import build_temporal_split
from wine_quality.shared.domain.forecast_dataset import ForecastDataset
from wine_quality.shared.domain.metrics import Metrics
from wine_quality.shared.domain.types import JsonObject


def _frame(days: int = 100) -> pl.DataFrame:
    start = date(2020, 1, 1)
    counts = np.cumsum(10 + np.arange(days) % 7 + np.arange(days) // 10)
    return pl.DataFrame(
        {
            "Province/State": ["A", "B"],
            "Country/Region": ["Example", "Example"],
            "Lat": [1.0, 2.0],
            "Long": [3.0, 4.0],
            **{
                (start + timedelta(days=i)).strftime("%m/%d/%y"): [int(n), int(n * 2)]
                for i, n in enumerate(counts)
            },
        }
    )


@pytest.fixture
def covid_dataset(tmp_path: Path) -> ForecastDataset:
    path = tmp_path / "covid.csv"
    _frame().write_csv(path)
    return CovidPreprocessingService(PolarsCovidRepository()).run(path, "Example")


def test_geographic_dedup_and_aggregation_preserve_original_ids(tmp_path: Path) -> None:
    frame = _frame()
    path = tmp_path / "covid.csv"
    pl.concat([frame, frame.head(1)]).write_csv(path)
    series = PolarsCovidRepository().load(path, "Example")
    assert series.rows_raw == 3
    assert series.duplicates_removed == 1
    np.testing.assert_array_equal(series.source_row_ids, [0, 1])
    assert series.cumulative[0] == 30
    assert not series.cumulative.flags.writeable
    with pytest.raises(ValueError, match="país"):
        PolarsCovidRepository().load(path, "Missing")


def test_lags_means_and_calendar_use_only_available_past(covid_dataset: ForecastDataset) -> None:
    changes = np.diff(covid_dataset.series.cumulative)
    assert covid_dataset.target.size == 85
    assert covid_dataset.features.shape == (85, 18)
    assert covid_dataset.target_dates[0] == date(2020, 1, 16)
    np.testing.assert_array_equal(covid_dataset.features[0, :14], changes[:14][::-1])
    assert covid_dataset.target[0] == changes[14]
    assert covid_dataset.features[0, 14] == changes[7:14].mean()
    assert covid_dataset.features[0, 15] == changes[:14].mean()
    assert not covid_dataset.features.flags.writeable


def test_changing_future_cannot_change_earlier_features(tmp_path: Path) -> None:
    frame = _frame()
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    frame.write_csv(first)
    frame.with_columns(pl.col(frame.columns[54:]) + 1000).write_csv(second)
    service = CovidPreprocessingService(PolarsCovidRepository())
    before, after = service.run(first, "Example"), service.run(second, "Example")
    np.testing.assert_array_equal(before.features[:36], after.features[:36])
    np.testing.assert_array_equal(before.target[:35], after.target[:35])
    assert before.target[35] != after.target[35]


def test_negative_revisions_are_retained(tmp_path: Path) -> None:
    frame = _frame()
    column = frame.columns[30]
    frame = frame.with_columns(pl.lit(0).alias(column))
    path = tmp_path / "revisions.csv"
    frame.write_csv(path)
    dataset = CovidPreprocessingService(PolarsCovidRepository()).run(path, "Example")
    assert np.any(dataset.target < 0)
    assert series_summary(dataset.series)["negative_revision_count"] == 1


@pytest.mark.parametrize("invalid", [None, -1.0, 1.5, float("inf"), float("nan")])
def test_invalid_cumulative_counts_fail(tmp_path: Path, invalid: float | None) -> None:
    frame = _frame()
    path = tmp_path / "invalid.csv"
    frame.with_columns(pl.lit(invalid).alias(frame.columns[20])).write_csv(path)
    with pytest.raises(ValueError, match="acumulados"):
        PolarsCovidRepository().load(path, "Example")


def test_irregular_dates_and_conflicting_geography_fail(tmp_path: Path) -> None:
    frame = _frame()
    path = tmp_path / "bad.csv"
    frame.drop(frame.columns[20]).write_csv(path)
    with pytest.raises(ValueError, match="consecutivas"):
        PolarsCovidRepository().load(path, "Example")
    conflict = frame.head(1).with_columns(pl.lit(0, dtype=pl.Int64).alias(frame.columns[20]))
    pl.concat([frame, conflict]).write_csv(path)
    with pytest.raises(ValueError, match="geográficas"):
        PolarsCovidRepository().load(path, "Example")


def test_holdout_and_folds_respect_chronology(covid_dataset: ForecastDataset) -> None:
    split = build_temporal_split(covid_dataset, CovidConfig())
    assert split.train_indices.size == 68
    assert split.test_indices.size == 17
    assert split.train_indices[-1] < split.test_indices[0]
    previous_train_size = 0
    for fit, validation in split.cv_folds:
        assert fit[-1] < validation[0]
        assert validation[-1] < split.test_indices[0]
        assert fit.size > previous_train_size
        previous_train_size = fit.size


def test_both_scalers_fit_only_each_fold(covid_dataset: ForecastDataset) -> None:
    split = build_temporal_split(covid_dataset, CovidConfig())
    features, target = (
        covid_dataset.features[split.train_indices],
        covid_dataset.target[split.train_indices],
    )
    result = cross_validate(
        covid_model_catalog(42)[1].build(),
        features,
        target,
        cv=list(split.cv_folds),
        return_estimator=True,
    )
    for model, (fit, _) in zip(result["estimator"], split.cv_folds, strict=True):
        np.testing.assert_allclose(model.transformer_.mean_, [target[fit].mean()])
        np.testing.assert_allclose(
            model.regressor_.named_steps["scaler"].mean_, features[fit].mean(axis=0)
        )


def test_selection_and_baselines_are_traceable(covid_dataset: ForecastDataset) -> None:
    report = CovidExperimentService(covid_model_catalog(42)[:2]).run(covid_dataset, CovidConfig())
    json.dumps(report, allow_nan=False)
    models = cast(list[JsonObject], report["models"])
    selected = min(models, key=lambda m: cast(float, cast(JsonObject, m["cv"])["rmse_mean"]))
    assert report["winner_id"] == selected["id"]
    assert all("test" not in model for model in models)
    test = cast(JsonObject, report["test"])
    assert set(test) == {"winner", "persistence_1", "persistence_7"}
    plan = build_temporal_split(covid_dataset, CovidConfig())
    for key, raw in test.items():
        result = cast(JsonObject, raw)
        rows = cast(list[JsonObject], result["predictions"])
        actual = np.array([row["actual"] for row in rows], dtype=np.float64)
        predicted = np.array([row["predicted"] for row in rows], dtype=np.float64)
        assert Metrics.calculate(actual, predicted).as_json() == result["metrics"]
        if key != "winner":
            lag = 1 if key == "persistence_1" else 7
            np.testing.assert_array_equal(
                predicted, covid_dataset.features[plan.test_indices, lag - 1]
            )
