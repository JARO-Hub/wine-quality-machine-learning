import csv
import io
from datetime import datetime, timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np
import polars as pl

from wine_quality.shared.domain.covid_series import CovidSeries

GEOGRAPHIC_COLUMNS = ("Province/State", "Country/Region", "Lat", "Long")


class PolarsCovidRepository:
    def load(self, path: Path, country: str) -> CovidSeries:
        content = path.read_bytes()
        if not content.strip():
            raise ValueError("El CSV de COVID está vacío.")
        header = next(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
        if tuple(header[:4]) != GEOGRAPHIC_COLUMNS or len(header) != len(set(header)):
            raise ValueError("Esquema COVID incorrecto o columnas repetidas.")
        try:
            dates = tuple(datetime.strptime(name, "%m/%d/%y").date() for name in header[4:])
        except ValueError as error:
            raise ValueError("Las columnas temporales deben ser fechas mes/día/año.") from error
        if len(dates) < 30 or any(
            right - left != timedelta(days=1) for left, right in zip(dates, dates[1:], strict=False)
        ):
            raise ValueError("Se requieren al menos 30 fechas diarias consecutivas y ordenadas.")
        frame = pl.read_csv(content, schema_overrides=dict.fromkeys(GEOGRAPHIC_COLUMNS, pl.String))
        if not frame.height:
            raise ValueError("El CSV de COVID no tiene filas geográficas.")
        counts = frame.select(header[4:]).cast(pl.Float64, strict=True)
        values = counts.to_numpy()
        if (
            any(counts.null_count().row(0))
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values != np.floor(values))
        ):
            raise ValueError(
                "Los acumulados deben ser enteros finitos no negativos, sin faltantes."
            )
        if (
            frame["Country/Region"].null_count()
            or frame.filter(pl.col("Country/Region").str.strip_chars() == "").height
        ):
            raise ValueError("Todas las filas necesitan un país o región.")
        unique = frame.with_row_index("source_row_id").unique(
            subset=frame.columns, keep="first", maintain_order=True
        )
        selected = unique.filter(pl.col("Country/Region") == country)
        if not selected.height:
            raise ValueError(f"No se encontró el país {country!r}; usa su nombre exacto en el CSV.")
        if selected.select("Province/State", "Country/Region").is_duplicated().any():
            raise ValueError(
                "Hay filas geográficas repetidas con valores distintos; revisar origen."
            )
        cumulative = (
            selected.select(header[4:]).cast(pl.Float64).sum().to_numpy().ravel().astype(np.float64)
        )
        return CovidSeries(
            country=country,
            dates=dates,
            cumulative=cumulative,
            source_row_ids=selected["source_row_id"].to_numpy().astype(np.int64),
            rows_raw=frame.height,
            duplicates_removed=frame.height - unique.height,
            country_count=frame["Country/Region"].n_unique(),
            input_sha256=sha256(content).hexdigest(),
        )
