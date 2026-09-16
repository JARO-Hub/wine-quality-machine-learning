import argparse
import json
from pathlib import Path
from typing import cast

import polars as pl

from wine_quality.cases.case_01_preprocessing.service import PreprocessingService
from wine_quality.cases.case_02_model_selection.experiment_config import ExperimentConfig
from wine_quality.cases.case_02_model_selection.service import ModelSelectionService
from wine_quality.shared.adapters.polars_dataset_repository import PolarsDatasetRepository
from wine_quality.shared.adapters.sklearn_evaluation_engine import (
    SklearnEvaluationEngine,
    dataset_summary,
)
from wine_quality.shared.adapters.sklearn_model_catalog import model_catalog
from wine_quality.shared.domain.dataset import Dataset
from wine_quality.shared.domain.types import JsonObject


def _save_json(path: Path, report: JsonObject) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _print_selection(report: JsonObject) -> None:
    print(f"{'Modelo':26} {'RMSE CV':>9} {'DE CV':>9} {'MAE CV':>9} {'RMSE train':>11}")
    for model in cast(list[JsonObject], report["models"]):
        cv = cast(JsonObject, model["cv"])
        train = cast(JsonObject, model["train"])
        print(
            f"{str(model['label']):26} {cast(float, cv['rmse_mean']):9.4f} "
            f"{cast(float, cv['rmse_std']):9.4f} {cast(float, cv['mae_mean']):9.4f} "
            f"{cast(float, train['rmse']):11.4f}"
        )
    print(f"\nGanador elegido por validación: {report['winner_id']}")
    test = cast(JsonObject, report["test"])
    for name in ("winner", "baseline"):
        result = cast(JsonObject, test[name])
        metrics = cast(JsonObject, result["metrics"])
        r2 = metrics["r2"]
        r2_text = "indefinido" if r2 is None else f"{cast(float, r2):.4f}"
        print(
            f"Prueba {str(result['id']):14}: RMSE={cast(float, metrics['rmse']):.4f}; "
            f"MAE={cast(float, metrics['mae']):.4f}; "
            f"MSE={cast(float, metrics['mse']):.4f}; R²={r2_text}"
        )


def _export_clean(dataset: Dataset, directory: Path) -> None:
    frame = pl.DataFrame(dataset.features, schema=list(dataset.feature_names), orient="row")
    frame.with_columns(pl.Series("quality", dataset.target).cast(pl.Int64)).write_csv(
        directory / "clean.csv"
    )
    audit = dataset_summary(dataset)
    audit.update(
        {
            "input_sha256": dataset.input_sha256,
            "feature_names": list(dataset.feature_names),
            "kept_original_row_ids": [int(row_id) for row_id in dataset.row_ids],
            "policy": "exact_dedup_only; no_capping; no_scaling_before_split",
        }
    )
    _save_json(directory / "report.json", audit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocesamiento y selección reproducible de modelos."
    )
    parser.add_argument("command", choices=("preprocess", "select-models"))
    parser.add_argument("--data", type=Path, required=True, help="CSV original de vino tinto.")
    parser.add_argument("--output", type=Path, help="Directorio de resultados.")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    output = cast(Path | None, arguments.output) or Path(
        "outputs/01_preprocessing"
        if arguments.command == "preprocess"
        else "outputs/02_model_selection"
    )
    try:
        dataset = PreprocessingService(PolarsDatasetRepository()).run(arguments.data)
        output.mkdir(parents=True, exist_ok=True)
        print(
            f"Datos: {dataset.rows_raw} originales; {dataset.duplicates_removed} duplicados; "
            f"{dataset.rows_clean} únicos; {len(dataset.feature_names)} predictores."
        )
        if arguments.command == "preprocess":
            _export_clean(dataset, output)
        else:
            config = ExperimentConfig(seed=arguments.seed)
            service = ModelSelectionService(SklearnEvaluationEngine(), model_catalog(config.seed))
            report = service.run(dataset, config)
            _save_json(output / "report.json", report)
            _print_selection(report)
        print(f"Resultados: {output.resolve()}")
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        parser.exit(status=2, message=f"Error: {error}\n")
