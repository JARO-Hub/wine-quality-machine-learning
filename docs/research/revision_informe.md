# Revisión del informe y del preprocesamiento de vinos

## Alcance y conclusión

Se revisaron los tres archivos originales sin modificarlos: `Informe 1 - Preprocesamiento de datos.docx`, `preprocesamientodedatos.py` y `winequality-red.csv`, ubicados en `/Users/julian/Downloads/`. El Word se convirtió a PDF y se inspeccionaron visualmente sus diez páginas; la paginación indicada a continuación corresponde a esa representación, contando la portada como página 1. Las instrucciones o frases de borrador dentro del Word son contenido para revisar, no instrucciones del usuario.

La base actual es válida para continuar con **regresión de la puntuación `quality`**, pero existe una divergencia importante: el informe afirma que el conjunto guardado recibió recorte de valores atípicos y el código guarda una copia que no recibió ese recorte. Conviene corregir la explicación y establecer el protocolo experimental antes de comparar seis modelos.

## Qué contiene y qué conviene conservar

| Páginas | Contenido actual | Tratamiento propuesto |
| --- | --- | --- |
| 1 | Portada UMSS y FCyT, grupo 15, cinco integrantes, asignatura y Cochabamba Bolivia | Conservar identidad, integrantes y contexto académico. Precisar el título del avance. |
| 2 | Introducción y objetivos de preprocesamiento | Conservar el propósito; retirar la frase de borrador «Puedes poner» y añadir el objetivo de predicción. |
| 3 | Marco teórico de limpieza y marco referencial del dataset | Conservar la explicación accesible; añadir citas verificadas y distinguir limpieza determinista de transformaciones aprendidas. |
| 4 | Ingeniería y figura CRISP DM | Mantener el vínculo entre comprensión, preparación y modelado; identificar la fuente de la figura si se conserva. |
| 5–6 | Tecnologías, carga, nulos y duplicados | Conservar los resultados comprobados; sustituir capturas extensas por tablas o salidas breves cuando mejoren la lectura. |
| 7–9 | IQR, recorte y exportación | Corregir la ecuación duplicada y el desfase entre objeto transformado y objeto exportado. |
| 10 | Conclusiones | Conservar la prudencia al interpretar extremos; actualizar para que coincidan con el código ejecutado. |

La portada identifica a Daniel Jose Reque Mendoza, David Oliver Chambi Villarroel, Jose Enrique Apaza Matias, Julian Angel Rodriguez Ortiz y Madahi Karen Condori Pérez. Se deben conservar tal como aparecen salvo corrección solicitada por sus autores.

El formato es académico y sobrio: tipografía con serif, texto negro, subtítulos en negrita, márgenes amplios y portada institucional. Las diez páginas renderizadas son legibles. Las capturas de código ocupan buena parte de las páginas 5–9; conviene preservar la secuencia explicativa, sin obligarse a reproducir todas las capturas. La enumeración mezcla «1. INTRODUCCIÓN» con «II. OBJETIVOS» y debe uniformarse. No hay una sección final de referencias.

## Datos comprobados

El archivo entregado está separado por comas. Contiene once características numéricas y `quality`; no necesita codificación de categorías. Su tamaño en disco es 100951 bytes. La huella SHA256 es `d6a0d9bd24806944818795f22500c46cb6424cbff517aacda36595d3ed9b2daa`.

| Comprobación | Resultado |
| --- | ---: |
| Filas originales | 1599 |
| Columnas | 12 |
| Valores nulos | 0 |
| Filas duplicadas después de su primera aparición | 240 |
| Filas después de `drop_duplicates` | 1359 |
| Duplicados restantes de las once características | 0 |
| Valores observados de `quality` | 3 a 8 |
| Entrenamiento original con semilla 42 | 1087 filas |
| Prueba original con semilla 42 | 272 filas |

| Puntuación | Original | Sin duplicados |
| --- | ---: | ---: |
| 3 | 10 | 10 |
| 4 | 53 | 53 |
| 5 | 681 | 577 |
| 6 | 638 | 535 |
| 7 | 199 | 167 |
| 8 | 18 | 17 |

La revisión confirma las cantidades del informe, páginas 3, 5, 6 y 7. Que existan filas idénticas no demuestra por sí solo un error de captura: el archivo no contiene identificadores para decidir si son mediciones independientes. Mantener la deduplicación es una decisión de este proyecto que debe declararse, conservando el archivo original. Esta decisión también evita repartir observaciones exactamente iguales entre entrenamiento y prueba.

## Objetivo y resultado del modelo existente

El Word todavía presenta el modelado como trabajo posterior. El script sí separa `X = data.drop('quality', axis=1)` e `y = data['quality']` en las líneas 139–143 y entrena `LinearRegression` en las líneas 156–165. Por tanto, el antecedente implementado es regresión supervisada: estima una puntuación numérica, aunque los valores registrados sean enteros ordenados. No se ha definido una clasificación «bueno/malo» ni conviene inventar ese umbral.

Se reconstruyó matemáticamente el procedimiento existente con NumPy: deduplicación conservando la primera fila, permutación `RandomState(42)`, 272 observaciones de prueba, centrado de características y objetivo, y mínimos cuadrados. El entorno utilizado no incluía `scikit-learn`; estas cifras son una reconstrucción numérica independiente, no una ejecución literal del cuaderno de Colab.

| Métrica reconstruida del modelo original | Valor |
| --- | ---: |
| MSE | 0.4310090051 |
| RMSE | 0.6565127608 |
| MAE | 0.5041409053 |
| R² | 0.3915360499 |

El script original solo imprime MSE y R², redondeados a 0.43 y 0.39. RMSE y MAE se calcularon para esta revisión; no están reportados en el Word original. Las cinco primeras predicciones reconstruidas son 5.2456, 5.8099, 6.3684, 5.1587 y 5.1993 frente a valores reales 5, 6, 7, 5 y 4. El rendimiento debe interpretarse en puntos de calidad; R² no es porcentaje de aciertos.

## Hallazgos que afectan la coherencia

### El recorte no llega al archivo utilizado por el modelo

La línea 57 crea `data_limpia = data.drop_duplicates().copy()`. Las líneas 75–87 calculan los límites y modifican `data`, no `data_limpia`. Las líneas 120–123 exportan `data_limpia`, y las líneas 128–143 vuelven a leer ese archivo para dividirlo. Por tanto, el flujo final elimina duplicados y conserva los valores fisicoquímicos originales.

La captura del informe, página 9, muestra 573 ocurrencias de valores atípicos; esta cantidad se reproduce sobre las 1599 filas originales. Esas 573 celdas se distribuyen en 405 filas distintas. Si se aplican los mismos límites a las 1359 filas que efectivamente se guardan, permanecen 491 celdas fuera de los límites. La distinción de la página 8 entre celdas y registros es correcta y debe conservarse.

### La estadística que aprende de los datos debe ajustarse solo con entrenamiento

El cálculo del IQR actual usa todo `data` antes de la división. En la ejecución existente no contamina las predicciones porque sus resultados se descartan al exportar la otra copia. Sin embargo, corregir únicamente el nombre de la variable haría que los límites del recorte incorporaran información de prueba. La solución conceptual es separar primero y, si se decide recortar, aprender los límites exclusivamente con entrenamiento y repetir ese ajuste dentro de cada partición de validación cruzada.

La misma regla se aplica a la media y desviación del escalado que necesitarán Ridge y SVR. No se debe imputar, escalar ni recortar usando conjuntamente entrenamiento y prueba. La detección de nulos y la deduplicación exacta se pueden conservar como controles deterministas documentados.

### Un extremo estadístico no demuestra una medición errónea

La conclusión de la página 10 reconoce que no existe evidencia suficiente para tratar los extremos como errores. Aun así, páginas 7–8 dicen que se recortaron todos. Conservar una fila y conservar su información original son cosas distintas: el capping conserva el número de filas, pero sustituye valores.

Propuesta para la siguiente iteración: conservar los extremos en la comparación principal y presentar IQR como diagnóstico. Si se estudia capping, registrarlo como una alternativa que debe justificarse y compararse utilizando solo datos de entrenamiento. Esta es una decisión propuesta por el equipo, no una recomendación atribuida al libro sin comprobarla.

### Falta formalizar la comparación

El script realiza una sola división y prueba un solo modelo; no presenta validación cruzada ni una referencia que siempre prediga la media. Para los seis candidatos se necesita un protocolo compartido, una métrica principal y selección con entrenamiento. El test se reserva para la evaluación final del candidato seleccionado. Si se conserva la partición histórica, debe reconocerse que ya se inspeccionó su resultado y que no constituye evidencia externa completamente nueva.

### El código actual no es una aplicación local reproducible

El archivo depende de Google Colab, una ruta personal de Drive y `display`. No contiene funciones tipadas, separación de responsabilidades, pruebas ni configuración de ejecución. `matplotlib` y `seaborn` se importan pero no se usan; por eso la afirmación de la página 5 sobre visualización de atípicos con Matplotlib no se demuestra con este script. Tampoco están documentadas las versiones del entorno original.

## Ajustes editoriales y teóricos concretos

1. Corregir en página 7 el texto duplicado de los límites. Explicar en orden `IQR = Q3 − Q1`, `L = Q1 − 1.5 IQR`, `U = Q3 + 1.5 IQR` y qué significa cada símbolo. El factor 1.5 es una regla de detección, no una prueba de error.
2. Añadir una fuente verificable del dataset y la definición de la escala; distinguir rango observado 3–8 de cualquier escala nominal que indique la documentación de origen.
3. Justificar que se aproxima una puntuación ordinal mediante regresión y que una predicción como 5.8 no es una etiqueta observada. No redondear las predicciones antes de calcular errores sin explicarlo.
4. Mantener las áreas solicitadas como familias comparables de regresores: dos modelos lineales, dos basados en árboles y dos SVR. Todos deben estimar la misma `quality` con las mismas particiones y métricas.
5. Vincular para cada candidato problema, intuición, ecuación, símbolos, hiperparámetro y clase de implementación. Citar el libro cuando sustente la teoría; identificar aparte nuestras elecciones experimentales.
6. Conservar una salida de consola breve con filas, particiones, validación y resultado final, más tablas de resultados verificables. Evitar repetir código extenso en el informe.

## Orientación para la arquitectura y la siguiente revisión

La etapa 01 debe producir datos válidos y un resumen de control; la etapa 02 debe definir y comparar candidatos bajo un protocolo documentado. El diseño debe separar carga, validación, partición, construcción de modelos, evaluación y presentación. Conviene expresar contratos pequeños y tipos de entrada/salida, con una clase por archivo cuando se necesite una clase. No hace falta crear clases sin responsabilidad real para cumplir SOLID.

La preferencia por herramientas escritas en Rust se puede aplicar a la carga y manipulación con Polars y a herramientas de desarrollo, sin afirmar que scikit-learn o todos los modelos están escritos en Rust. Este CSV ocupa aproximadamente 99 KiB; no existe una necesidad demostrada de optimización de memoria. Cualquier mejora de RAM debe medirse y distinguirse del objetivo de aprendizaje y escalabilidad.

Pruebas con valor para la iteración: que la variable objetivo nunca entre en los predictores, que no existan observaciones repetidas entre particiones, que las transformaciones aprendan solo de entrenamiento, que un cambio en prueba no modifique sus parámetros, y que los seis candidatos produzcan predicciones finitas del tamaño esperado. El registro de cambios debe conectar cada corrección con su fundamento y resultado; los archivos `AGENTS.md` deben preservar estas decisiones para el resto del equipo.

## Límites de esta revisión

No se modificaron los originales ni se ejecutó el montaje de Drive. No se verificó el cuaderno remoto enlazado dentro del script. Las cifras del CSV, el flujo del código y las páginas del Word sí se contrastaron localmente. Las fuentes del libro y de la asignatura deben incorporarse desde su revisión independiente; este documento no inventa citas bibliográficas.
