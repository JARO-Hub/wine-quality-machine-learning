from dataclasses import dataclass
from datetime import date

from wine_quality.shared.domain.covid_series import CovidSeries
from wine_quality.shared.domain.types import FloatArray, IndexArray


@dataclass(frozen=True, slots=True)
class ForecastDataset:
    series: CovidSeries
    features: FloatArray
    target: FloatArray
    target_dates: tuple[date, ...]
    target_date_indices: IndexArray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        for values in (self.features, self.target, self.target_date_indices):
            values.setflags(write=False)
