from dataclasses import dataclass

from wine_quality.shared.domain.types import IndexArray


@dataclass(frozen=True, slots=True)
class SplitPlan:
    train_indices: IndexArray
    test_indices: IndexArray
    cv_folds: tuple[tuple[IndexArray, IndexArray], ...]
