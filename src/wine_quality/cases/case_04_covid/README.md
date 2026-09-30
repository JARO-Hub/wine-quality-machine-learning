# Caso 04: COVID y validación temporal

Entrada: CSV JHU de confirmados acumulados, con una columna por fecha.
Salida: serie nacional, tabla supervisada, comparación temporal y prueba del
ganador con persistencia de uno y siete días. El país inicial es Bolivia.
Leer [el protocolo](../../../../docs/decisions/004-covid-temporal.md) antes de entrenar.

```powershell
uv sync --locked --extra dev
uv run covid-quality preprocess
uv run covid-quality run
uv run python scripts/build_dashboard.py --report outputs/04_covid/report.json --output outputs/04_covid/results.html
```

En equipos sin AVX2, añade `--extra compat` a `uv sync` y usa
`uv run --extra compat covid-quality run`; ese extra mantiene la misma versión
de Polars y selecciona su runtime compatible. Para otro país y evidencia separada:

```powershell
uv run covid-quality run --country Argentina --output outputs/covid_argentina
```

`preprocessing.py` convierte acumulados en variaciones y crea 14 retardos, dos
medias pasadas y dos términos de calendario. `config.py` fija el experimento.
`service.py` reutiliza `tune_model` con folds temporales; el catálogo compartido
se adapta con escalado de respuesta y una RNA nueva 18→8→1.
Los repositorios y estructuras están en `shared/`.

No usar el motor estratificado de vinos para COVID. Los parámetros permanecen
fijos durante prueba, pero cada entrada usa registros reales de días anteriores:
es pronóstico secuencial a un día, no predicción simultánea de todo el bloque.
Las revisiones negativas se conservan; el primer cambio diario es indefinido.
La copia histórica no prueba que los valores estuvieran disponibles sin revisiones
en la fecha original. Ninguna consulta representa una predicción actual de 2026.

El informe se reconstruye exclusivamente desde evidencia persistida:

```powershell
uv sync --locked --extra dev --extra reports
uv run --extra reports python scripts/build_covid_report.py
```

Esta herramienta genera Markdown, Word y figuras; no entrena ni recalcula métricas.
Después de generar Word, renderizar y revisar sus páginas antes de entregarlo.
