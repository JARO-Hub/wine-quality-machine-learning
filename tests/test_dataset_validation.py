from pathlib import Path

import numpy as np
import polars as pl
import pytest

from wine_quality.shared.adapters.polars_dataset_repository import PolarsDatasetRepository


def test_empty_csv_has_clear_failure(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="vacío"):
        PolarsDatasetRepository().load(path)


def test_exact_dedup_preserves_first_rows_and_extremes(
    tmp_path: Path, synthetic_frame: pl.DataFrame
) -> None:
    original = synthetic_frame.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(99.0)
        .otherwise(pl.col("alcohol"))
        .alias("alcohol")
    )
    path = tmp_path / "duplicates.csv"
    pl.concat([original, original.head(2)]).write_csv(path)
    result = PolarsDatasetRepository().load(path)
    assert result.rows_raw == 62
    assert result.rows_clean == 60
    assert result.duplicates_removed == 2
    np.testing.assert_array_equal(result.row_ids, np.arange(60))
    assert result.features[0, 10] == 99.0
    assert result.features.flags.c_contiguous
    assert not result.features.flags.writeable


@pytest.mark.parametrize("separator", [",", ";"])
def test_reads_real_and_uci_delimiters(
    separator: str, tmp_path: Path, synthetic_frame: pl.DataFrame
) -> None:
    path = tmp_path / "separator.csv"
    synthetic_frame.write_csv(path, separator=separator)
    assert PolarsDatasetRepository().load(path).features.shape == (60, 11)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, None])
def test_rejects_invalid_predictors(
    invalid: float | None, tmp_path: Path, synthetic_frame: pl.DataFrame
) -> None:
    path = tmp_path / "invalid.csv"
    synthetic_frame.with_columns(pl.lit(invalid).alias("alcohol")).write_csv(path)
    with pytest.raises(ValueError):
        PolarsDatasetRepository().load(path)


@pytest.mark.parametrize("invalid", [11.0, -1.0, 5.5])
def test_rejects_invalid_target(
    invalid: float, tmp_path: Path, synthetic_frame: pl.DataFrame
) -> None:
    path = tmp_path / "invalid_target.csv"
    synthetic_frame.with_columns(pl.lit(invalid).alias("quality")).write_csv(path)
    with pytest.raises(ValueError, match="quality"):
        PolarsDatasetRepository().load(path)


def test_conflicting_targets_cannot_cross_splits(
    tmp_path: Path, synthetic_frame: pl.DataFrame
) -> None:
    conflict = synthetic_frame.head(1).with_columns(pl.lit(8).cast(pl.Int64).alias("quality"))
    path = tmp_path / "conflict.csv"
    pl.concat([synthetic_frame, conflict]).write_csv(path)
    with pytest.raises(ValueError, match="grupos"):
        PolarsDatasetRepository().load(path)


def test_target_cannot_silently_become_a_feature(
    tmp_path: Path, synthetic_frame: pl.DataFrame
) -> None:
    path = tmp_path / "bad_schema.csv"
    synthetic_frame.rename({"quality": "quality_copy"}).write_csv(path)
    with pytest.raises(ValueError, match="Esquema"):
        PolarsDatasetRepository().load(path)
