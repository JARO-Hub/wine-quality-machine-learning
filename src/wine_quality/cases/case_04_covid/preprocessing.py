from pathlib import Path

import numpy as np

from wine_quality.shared.domain.covid_series import CovidSeries
from wine_quality.shared.domain.forecast_dataset import ForecastDataset
from wine_quality.shared.domain.types import JsonObject
from wine_quality.shared.ports.covid_repository import CovidRepository

LAG_COUNT = 14
FEATURE_NAMES = (
    *(f"daily_lag_{lag}" for lag in range(1, LAG_COUNT + 1)),
    "daily_mean_7",
    "daily_mean_14",
    "weekday_sin",
    "weekday_cos",
)


def series_summary(series: CovidSeries) -> JsonObject:
    changes = np.diff(series.cumulative)
    return {
        "country": series.country,
        "rows_raw": series.rows_raw,
        "duplicates_removed": series.duplicates_removed,
        "countries_in_csv": series.country_count,
        "source_row_ids": [int(index) for index in series.source_row_ids],
        "date_count": len(series.dates),
        "first_date": series.dates[0].isoformat(),
        "last_date": series.dates[-1].isoformat(),
        "last_cumulative": float(series.cumulative[-1]),
        "negative_revision_count": int(np.count_nonzero(changes < 0)),
        "negative_revisions": [
            {"date": series.dates[i + 1].isoformat(), "change": float(changes[i])}
            for i in np.flatnonzero(changes < 0)
        ],
        "input_sha256": series.input_sha256,
    }


class CovidPreprocessingService:
    def __init__(self, repository: CovidRepository) -> None:
        self._repository = repository

    def run(self, path: Path, country: str) -> ForecastDataset:
        series = self._repository.load(path, country)
        changes = np.diff(series.cumulative)
        features: list[list[float]] = []
        for i in range(LAG_COUNT, changes.size):
            weekday = 2 * np.pi * series.dates[i + 1].weekday() / 7
            features.append(
                [float(changes[i - lag]) for lag in range(1, LAG_COUNT + 1)]
                + [
                    float(changes[i - 7 : i].mean()),
                    float(changes[i - LAG_COUNT : i].mean()),
                    float(np.sin(weekday)),
                    float(np.cos(weekday)),
                ]
            )
        return ForecastDataset(
            series=series,
            features=np.asarray(features, dtype=np.float64),
            target=changes[LAG_COUNT:].copy(),
            target_dates=series.dates[LAG_COUNT + 1 :],
            target_date_indices=np.arange(LAG_COUNT + 1, len(series.dates), dtype=np.int64),
            feature_names=FEATURE_NAMES,
        )
