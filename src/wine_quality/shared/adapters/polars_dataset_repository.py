from hashlib import sha256
from pathlib import Path

import numpy as np
import polars as pl

from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.domain.schema import FEATURE_NAMES, TARGET_NAME


class PolarsDatasetRepository:
    def load(self, path: Path) -> Dataset:
        content = path.read_bytes()
        if not content.strip():
            raise ValueError("El archivo CSV está vacío.")
        header = content.splitlines()[0].decode("utf-8-sig")
        separator = ";" if header.count(";") > header.count(",") else ","
        frame = pl.read_csv(content, separator=separator)
        expected = (*FEATURE_NAMES, TARGET_NAME)
        if set(frame.columns) != set(expected):
            raise ValueError(f"Esquema incorrecto: se requieren exactamente {expected}.")
        frame = frame.select(expected).cast(pl.Float64, strict=True)
        if frame.height < 20:
            raise ValueError("Se requieren al menos 20 filas para este experimento.")
        if any(frame.null_count().row(0)):
            raise ValueError("Se encontraron valores faltantes; debe definirse su tratamiento.")
        values = frame.to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("Se encontraron valores infinitos o no numéricos.")
        quality = values[:, -1]
        if np.any(quality != np.floor(quality)) or np.any((quality < 0) | (quality > 10)):
            raise ValueError("quality debe ser un entero entre 0 y 10.")
        if np.any(values[:, :-1] < 0) or np.any(values[:, 8] > 14):
            raise ValueError("Predictores fuera de dominio: negativos o pH mayor que 14.")
        unique = frame.with_row_index("row_id").unique(
            subset=list(expected), keep="first", maintain_order=True
        )
        if unique.select(FEATURE_NAMES).is_duplicated().any():
            raise ValueError(
                "Predictores idénticos tienen distintos quality. Definir grupos antes de dividir."
            )
        features = np.asarray(unique.select(FEATURE_NAMES).to_numpy(), dtype=np.float64, order="C")
        target = np.asarray(unique[TARGET_NAME].to_numpy(), dtype=np.float64)
        row_ids = np.asarray(unique["row_id"].to_numpy(), dtype=np.int64)
        for array in (features, target, row_ids):
            array.setflags(write=False)
        return Dataset(
            features, target, row_ids, FEATURE_NAMES, frame.height, sha256(content).hexdigest()
        )
