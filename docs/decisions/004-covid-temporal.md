# Decisión 004: COVID como experimento temporal independiente

Fecha: 30 de septiembre de 2026. Estado: protocolo fijado antes de entrenar.

## Pregunta y alcance

¿Qué modelo estima mejor la variación de casos confirmados registrados del día
siguiente en Bolivia, usando únicamente registros anteriores y el calendario?
El país es configurable, pero cada país y CSV define un experimento separado.
Bolivia es nuestra elección inicial, no una recomendación de la fuente.
Los datos de vinos, sus particiones y resultados permanecen independientes.

El CSV aportado tiene 276 filas geográficas, 193 etiquetas de país/región y
494 fechas consecutivas, del 2020-01-22 al 2021-05-29. Se conserva intacto.
Sus columnas coinciden con el formato global JHU CSSE; esto identifica el formato,
no demuestra igualdad byte por byte con una revisión del repositorio remoto.
La atribución y las condiciones CC BY 4.0 están documentadas en
[el README de JHU, apartado Terms of Use](https://github.com/CSSEGISandData/COVID-19#terms-of-use).

## Respuesta y predictores

Sea C_t el total acumulado registrado para un país en la fecha t. Se suman las
filas de ese país después de quitar únicamente duplicados exactos conservando
sus identificadores originales. La respuesta es d_t = C_t - C_(t-1), la variación
diaria registrada; no equivale a infecciones ocurridas ese día. El primer día
carece de antecedente y no tiene diferencia definida. Una diferencia negativa
es una revisión del registro: se conserva, se cuenta y no se recorta.

Para estimar d_t al cierre de t-1, X_t contiene d_(t-1) hasta d_(t-14), las medias
de esos últimos 7 y 14 días, y seno/coseno del día de semana de t. Son 18 entradas.
El calendario futuro es conocido, pero C_t y d_t no participan en X_t.
Las medias usan ventanas pasadas sin centrar. Se mantienen ceros iniciales y
fechas distintas con valores idénticos; no son registros geográficos duplicados.
No se imputa, recorta, redondea ni aplica logaritmo a la respuesta.

## Evaluación fijada

Se reservan las últimas ceil(0.20 × n) muestras supervisadas para prueba.
El resto usa TimeSeriesSplit con cinco ventanas expansivas, sin barajar y gap=0.
Cada respuesta de ajuste precede a toda respuesta de validación. Gap=0 es válido
para este horizonte de un día: d_(t-1) se conoce al emitir el pronóstico de t.
No se pretende que las ventanas de características no compartan historia.
La fuente explica la validación cronológica; la fracción y las entradas son
decisiones de este proyecto, no parámetros prescritos por sus autores.
[Documentación TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

Se reutilizan las 43 configuraciones de OLS, Ridge, CART, bosque de 160 árboles,
SVR lineal y SVR RBF del catálogo de vinos. Se añade una RNA fija 18→8→1 tanh,
salida lineal, solver L-BFGS, alpha=0.001, max_iter=2000, max_fun=50000 y semilla 42.
Son 44 configuraciones. Esta RNA es una elección nueva para COVID, no el modelo
SGD 11→8→1 de vinos. Las entradas se estandarizan dentro del Pipeline de modelos
lineales, SVR y RNA. Todos los candidatos usan TransformedTargetRegressor con
StandardScaler para y, ajustado dentro de cada fold; la transformación inversa
devuelve predicciones en casos/día. Así C y epsilon de SVR se interpretan en
unidades de desviación de la respuesta de ajuste.
[Documentación de la transformación](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html).

Se elige la menor media de RMSE de validación entre los siete candidatos; los
empates exactos siguen el orden del catálogo. Las referencias no participan en
la elección: persistencia d_(t-1) y persistencia semanal d_(t-7). Se registran
sus métricas de validación y se evalúan en prueba junto al único ganador.
Los demás candidatos no tienen métricas de prueba.

El ganador se reajusta únicamente con entrenamiento. En prueba sus parámetros
permanecen fijos; las entradas de cada día incorporan los valores reales ya
observados de los días anteriores. Es evaluación secuencial a un día, no un
pronóstico de todo el bloque emitido desde una sola fecha. Una ejecución nueva
después de observar esta prueba debe reconocer su reutilización; no ajustar
modelos buscando mejorarla.

## Evidencia, unidades y límites

Se persisten SHA256 de bytes originales, país, filas originales, fechas,
características, folds, grids, parámetros completos, versiones, avisos, tiempos,
MAE, MSE, RMSE, R² y predicciones de prueba. nbytes mide buffers NumPy, no pico
de RAM. MAE/RMSE se expresan en casos registrados por día; MSE en su cuadrado.
R² no es porcentaje de aciertos. No se fabrica una matriz de confusión para
regresión. La dispersión entre folds no es un intervalo de confianza.

El README JHU, apartado Data modification records, documenta revisiones
históricas. Esta copia no contiene versiones tal como se publicaron cada día;
por eso el backtest cronológico usa una instantánea retrospectiva y no demuestra
disponibilidad real de todos sus valores en cada fecha.
[Registro de modificaciones JHU](https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/README.md#data-modification-records).
No se modelan población, pruebas, vacunas, intervenciones ni mortalidad.
Los resultados describen registros históricos y no sustentan decisiones sanitarias.
