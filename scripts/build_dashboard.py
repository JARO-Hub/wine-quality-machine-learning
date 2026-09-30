from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_dashboard(report_path: Path, output_path: Path) -> None:
    payload: object = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("Versión de report.json no compatible")
    template_path = Path(__file__).resolve().parents[1] / "templates" / "results.html"
    template = template_path.read_text(encoding="utf-8")
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template.replace("__REPORT_JSON__", serialized), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizacion de Datos Quicaso")
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/02_model_selection/report.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/02_model_selection/results.html")
    )
    arguments = parser.parse_args()
    build_dashboard(arguments.report, arguments.output)
    print(f"Visor: {arguments.output}")


if __name__ == "__main__":
    main()
