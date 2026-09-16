from pathlib import Path

import numpy as np
import polars as pl
import pytest

from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.domain.schema import FEATURE_NAMES


@pytest.fixture
def synthetic_frame() -> pl.DataFrame:
    features = {
        name: [float(1 + index * 0.01 + column * 0.1) for index in range(60)]
        for column, name in enumerate(FEATURE_NAMES)
    }
    return pl.DataFrame(features).with_columns(
        pl.Series("quality", [5 + index % 3 for index in range(60)])
    )


@pytest.fixture
def synthetic_csv(tmp_path: Path, synthetic_frame: pl.DataFrame) -> Path:
    path = tmp_path / "wine.csv"
    synthetic_frame.write_csv(path)
    return path


@pytest.fixture
def synthetic_dataset() -> Dataset:
    generator = np.random.default_rng(12)
    features = generator.normal(size=(100, 11)).astype(np.float64)
    target = np.asarray(5 + (features[:, 0] > 0) + (features[:, 1] > 0), dtype=np.float64)
    return Dataset(
        features, target, np.arange(100, dtype=np.int64), FEATURE_NAMES, 100, "synthetic"
    )
