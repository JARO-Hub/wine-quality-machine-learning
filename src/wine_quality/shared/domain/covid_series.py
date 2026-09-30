from dataclasses import dataclass
from datetime import date

from wine_quality.shared.domain.types import FloatArray, IndexArray


@dataclass(frozen=True, slots=True)
class CovidSeries:
    country: str
    dates: tuple[date, ...]
    cumulative: FloatArray
    source_row_ids: IndexArray
    rows_raw: int
    duplicates_removed: int
    country_count: int
    input_sha256: str

    def __post_init__(self) -> None:
        self.cumulative.setflags(write=False)
        self.source_row_ids.setflags(write=False)
