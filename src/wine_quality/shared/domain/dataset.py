from dataclasses import dataclass

from wine_quality.shared.domain.types import FloatArray, IndexArray


@dataclass(frozen=True, slots=True)
class Dataset:
    features: FloatArray
    target: FloatArray
    row_ids: IndexArray
    feature_names: tuple[str, ...]
    rows_raw: int
    input_sha256: str

    @property
    def rows_clean(self) -> int:
        return int(self.target.size)

    @property
    def duplicates_removed(self) -> int:
        return self.rows_raw - self.rows_clean
