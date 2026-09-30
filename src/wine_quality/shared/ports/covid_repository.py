from pathlib import Path
from typing import Protocol

from wine_quality.shared.domain.covid_series import CovidSeries


class CovidRepository(Protocol):
    def load(self, path: Path, country: str) -> CovidSeries: ...
