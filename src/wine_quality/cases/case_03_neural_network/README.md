# Caso 03 Red neuronal e ingreso manual

Pregunta: ¿cómo aprende y predice una RNA pequeña la calidad numérica de un vino?

Lee primero [la decisión 003](../../../../docs/decisions/003-red-neuronal.md) y [la guía de estudio](../../../../docs/research/guia_estudio_rna.md). Se conserva el CSV y todas las particiones del caso 02. Este caso recibe el motor compartido y un ModelSpec fijo; no duplica validación ni busca una arquitectura usando prueba.

## Ejecutar desde la raíz

```bash
uv sync --locked --extra dev
uv run wine-neural train
uv run wine-neural predict --trace
uv run wine-neural lesson
```

`train` guarda `outputs/03_neural_network/report.json` y `model.json`. La consulta interactiva pide las once mediciones y, opcionalmente, la calidad real. Usa `--output ruta.json` para guardar una consulta; `--values` permite pasar once números en el orden del CSV. `--quality` añade el error individual. Con `--trace` se observan entradas estandarizadas y activaciones ocultas.

```bash
uv run wine-neural predict --values 8.3 0.6 0.25 2.2 0.118 9 38 0.99616 3.15 0.53 9.8 --quality 5 --trace
```

Este ejemplo reproduce la fila histórica 1152, ya en prueba; no representa un nuevo vino ingresado por el usuario. La predicción registrada es 5.3897. Una entrada sin calidad conocida solo permite inferencia, no cálculo del error real.

## Qué corresponde a cada archivo

- `service.py`: coordina un candidato con el contrato de evaluación compartido.
- `lesson.py`: ejercicio 2→1→1; sus números no entrenan el dataset.
- `shared/adapters/sklearn_neural_network.py`: StandardScaler y MLPRegressor 11→8→1.
- `shared/domain/neural_network.py`: inferencia trazable con números guardados.
- `shared/adapters/neural_model_io.py`: persistencia JSON y comprobación de formato, columnas y dimensiones.
- `neural_cli.py`: argumentos, ingreso manual y presentación por consola.

El modelo exportado conserva pesos, sesgos, escalas y rangos de entrenamiento, sin objetos ejecutables. `predict` no modifica esos valores. La equivalencia con `Pipeline.predict` tiene una prueba específica.

## Resultado y límite

RNA fija: RMSE CV 0.6618, RMSE prueba histórica 0.6461. Los cinco folds y el reajuste final agotaron 1000 épocas sin confirmar convergencia; los seis avisos se guardan en `models[0].fit_warnings`. No cambiar parámetros buscando mejorar esa prueba ya vista. Los resultados del caso 02 permanecen históricos.
