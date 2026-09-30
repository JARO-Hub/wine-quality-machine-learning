import argparse
from pathlib import Path
from typing import cast

import polars as pl

from wine_quality.cases.case_04_covid.config import CovidConfig
from wine_quality.cases.case_04_covid.preprocessing import CovidPreprocessingService, series_summary
from wine_quality.cases.case_04_covid.service import CovidExperimentService
from wine_quality.cli import _save_json
from wine_quality.shared.adapters.polars_covid_repository import PolarsCovidRepository
from wine_quality.shared.adapters.sklearn_covid_catalog import covid_model_catalog
from wine_quality.shared.domain.forecast_dataset import ForecastDataset
from wine_quality.shared.domain.types import JsonObject


def _export(dataset: ForecastDataset, output: Path) -> None:
    pl.DataFrame(
        {
            "date": dataset.series.dates,
            "cumulative": dataset.series.cumulative,
        }
    ).with_columns(pl.col("cumulative").diff().alias("daily_change")).write_csv(
        output / "series.csv"
    )
    frame = pl.DataFrame(dataset.features, schema=list(dataset.feature_names), orient="row")
    frame.with_columns(
        pl.Series("target_date", dataset.target_dates),
        pl.Series("target_date_index", dataset.target_date_indices),
        pl.Series("daily_change", dataset.target),
    ).write_csv(output / "supervised.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="COVID: preparar y evaluar un pronóstico temporal."
    )
    parser.add_argument("command", choices=("preprocess", "run"))
    parser.add_argument(
        "--data", type=Path, default=Path("data/raw/time_series_covid_19_confirmed.csv")
    )
    parser.add_argument("--country", default="Bolivia", help="Nombre exacto del país en el CSV.")
    parser.add_argument("--output", type=Path, default=Path("outputs/04_covid"))
    arguments = parser.parse_args()
    try:
        if (arguments.output / "report.json").exists():
            raise ValueError(
                "Ya existe una evaluación en esa carpeta. Usa --output con una carpeta nueva "
                "y declara la reutilización de prueba si repites el experimento."
            )
        dataset = CovidPreprocessingService(PolarsCovidRepository()).run(
            arguments.data, arguments.country
        )
        arguments.output.mkdir(parents=True, exist_ok=True)
        _export(dataset, arguments.output)
        if arguments.command == "preprocess":
            report = series_summary(dataset.series)
            report["supervised_count"] = int(dataset.target.size)
            report["feature_names"] = list(dataset.feature_names)
            _save_json(arguments.output / "preprocessing.json", report)
        else:
            config = CovidConfig()
            print(
                f"COVID {arguments.country}: {dataset.target.size} muestras; "
                "cinco folds temporales.",
                flush=True,
            )
            report = CovidExperimentService(covid_model_catalog(config.seed)).run(dataset, config)
            _save_json(arguments.output / "report.json", report)
            for model in cast(list[JsonObject], report["models"]):
                cv = cast(JsonObject, model["cv"])
                print(f"{str(model['label']):25} RMSE CV={cast(float, cv['rmse_mean']):.2f}")
            print(f"Elegido por validación: {report['winner_id']}")
            for key, value in cast(JsonObject, report["test"]).items():
                metrics = cast(JsonObject, cast(JsonObject, value)["metrics"])
                print(
                    f"Prueba {key}: RMSE={cast(float, metrics['rmse']):.2f}; "
                    f"MAE={cast(float, metrics['mae']):.2f}"
                )
        print(f"Resultados: {arguments.output.resolve()}")
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
