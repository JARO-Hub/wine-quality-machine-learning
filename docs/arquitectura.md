# Arquitectura por casos de uso

El diseño separa decisiones del experimento, bibliotecas concretas y presentación. La interfaz pública de cada caso es `run`; la consola reúne sus dependencias explícitamente. La carpeta 03 podrá seguir el mismo patrón sin duplicar la carga, los tipos o las métricas.

## Recorrido de una ejecución

`cli.py` entrega una ruta a `PreprocessingService`, que recibe una implementación de `DatasetRepository`. `PolarsDatasetRepository` valida esquema, finitos, dominio y duplicados, y devuelve `Dataset` con arreglos de sólo lectura e identificadores de las filas originales.

`ModelSelectionService` recibe un catálogo de `ModelSpec` y un `EvaluationEngine`. `SklearnEvaluationEngine` construye las particiones, ejecuta búsquedas, selecciona por RMSE CV y evalúa ganador y referencia. Las métricas se calculan con `Metrics`. La consola guarda un JSON; la plantilla HTML sólo lo presenta.

| Principio SOLID | Responsabilidad concreta |
| --- | --- |
| Responsabilidad única | Repositorio carga y valida; catálogo declara candidatos; motor evalúa; CLI escribe y muestra. |
| Abierto a extensión | Añadir un `ModelSpec` permite evaluar otro estimador con el mismo motor, sin otra cadena de condicionales por modelo. |
| Sustitución | Un repositorio o motor alternativo debe respetar entradas, tipos, significado y excepciones del contrato. Los protocolos no garantizan por sí solos corrección estadística; las pruebas la comprueban. |
| Interfaces pequeñas | `DatasetRepository`, `Regressor` y `EvaluationEngine` exponen únicamente las operaciones que consume cada caso. |
| Dependencias explícitas | Los servicios reciben repositorio, motor y catálogo; no crean en secreto clientes, rutas personales ni estado global. |

Las entidades usan dataclasses inmutables. Los datos experimentales son atributos públicos de sólo lectura; las dependencias internas del servicio usan `_engine`, `_models` o `_repository`, siguiendo la convención de Python para uso interno. No se usa herencia protegida sin necesidad ni getters repetitivos. La biblioteca dinámica scikit-learn se encapsula en adaptadores; mypy usa un override sólo para sus imports sin stubs.

## De la ecuación al archivo

| Decisión o fórmula | Implementación |
| --- | --- |
| quality separado de once características | `shared/domain/schema.py` y `shared/adapters/polars_dataset_repository.py` |
| Split80/20 y KFold5 sobre entrenamiento | `shared/adapters/sklearn_split.py` |
| SSE y penalización Ridge | `shared/adapters/sklearn_model_catalog.py`, `Ridge(solver="svd")` |
| Hojas con medias y error cuadrático | Catálogo, `DecisionTreeRegressor(criterion="squared_error")` |
| Bootstrap y promedio de160 árboles | Catálogo, `RandomForestRegressor` |
| Pérdida epsilon-insensible y kernel | Catálogo, `SVR(kernel="linear")` o `SVR(kernel="rbf")` |
| Escala ajustada por fold | Pipeline construido en catálogo; ajuste dentro de `GridSearchCV` |
| MAE, MSE, RMSE y R² | `shared/domain/metrics.py` |
| Selección antes de consultar test | `shared/adapters/sklearn_evaluation_engine.py` |

Las pruebas contrastan el gradiente de Ridge, medias en hojas, promedio de estimadores del bosque y expansión por kernel. No son una reimplementación del optimizador, pero permiten detectar divergencias entre lo documentado y las bibliotecas utilizadas.

## Crear la iteración 03

1. Escribir una decisión en `docs/decisions/` con pregunta, motivo, datos permitidos y métrica. Declarar que la prueba de02 ya fue observada.
2. Crear `cases/case_03_nombre/` tomando como patrón la estructura de02: `__init__.py`, configuración si cambia y servicio. Copiar el patrón, no mantener una segunda copia de utilidades compartidas.
3. Reutilizar los contratos y adaptadores cuando su significado siga siendo válido. Si la pregunta es clasificación o temporal, definir nuevos contratos de resultados y partición en vez de simular que el protocolo de regresión basta.
4. Escribir pruebas del comportamiento nuevo, con una clase por archivo cuando sea necesaria. Incorporar el caso a la consola.
5. Guardar resultados en otro directorio, actualizar el informe y `CHANGELOG.md`, y revisar fuentes y ecuaciones antes de crear el commit.

El bosque usa160 árboles y los SVR una caché64MB en esta iteración. Si el catálogo cambia esas constantes, actualizar también la metadata del motor y el informe: la metadata actual describe el catálogo estándar de02. No interpretar `nbytes` como pico de memoria del proceso.

## Implementación de la etapa 03

El caso de RNA reutiliza `EvaluationEngine` y `SklearnEvaluationEngine`; `EvaluationResult` reúne informe y estimador ajustado. El servicio 02 sigue devolviendo su informe y el caso 03 utiliza además el estimador para exportar parámetros numéricos. No se repite un entrenamiento para guardar el modelo.

El adaptador `sklearn_neural_network.py` construye el candidato fijo y extrae matrices; `neural_model_io.py` verifica su formato al guardar y cargar JSON. El dominio `NeuralNetwork` conserva matrices de solo lectura y reproduce el recorrido de predicción. `wine_input.py` valida medidas de consola. La CLI nueva expone tres acciones y `lesson.py` mantiene el ejemplo pedagógico separado del modelo real. Estas responsabilidades permiten estudiar inferencia sin entrar en toda la biblioteca.

Cualquier futura etapa 04 debe plantear primero su pregunta y protocolo. El formato JSON 11→8→1 es deliberadamente específico: no añadir soporte genérico a arquitecturas no requeridas. Si cambia esa arquitectura, versionar el formato, su comprobación de dimensiones y la correspondencia con la teoría.

## Implementación de la etapa 04: COVID

La decisión 004 define una tarea temporal independiente. CovidSeries conserva
el agregado del país, las fechas y las filas geográficas de origen;
ForecastDataset guarda 18 entradas pasadas y sus fechas objetivo.
CovidPreprocessingService depende del contrato CovidRepository; Polars ejecuta
la validación y agregación. No se usa Dataset de vinos para fingir que las
unidades geográficas son observaciones supervisadas independientes.

build_temporal_split devuelve SplitPlan con prueba final y TimeSeriesSplit.
CovidExperimentService comparte tune_model con el motor de vinos para buscar
configuraciones y registrar avisos. Su propia política evalúa el ganador y
las referencias temporales, sin ejecutar el split estratificado de vinos.
covid_model_catalog adapta el catálogo existente mediante
TransformedTargetRegressor y añade una RNA 18→8→1 fija con L-BFGS.
El JSON de inferencia neuronal de vinos no se reutiliza para esta arquitectura.

covid_cli exporta evidencia separada y rechaza carpetas ya evaluadas.
build_dashboard selecciona la plantilla por versión de esquema; COVID utiliza
covid-1.0. build_covid_report genera Markdown, Word y figuras a partir de
resultados persistidos. Los extras compat y reports separan el runtime de
Polars para equipos sin AVX2 y las dependencias de autoría de documentos.
