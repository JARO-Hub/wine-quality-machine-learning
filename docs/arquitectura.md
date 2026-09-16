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
