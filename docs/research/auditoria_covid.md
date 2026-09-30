# Auditoría de coherencia del caso COVID

Fecha: 30 de septiembre de 2026. Primera ejecución del protocolo 004.

## Evidencia revisada

Se verificaron los bytes del CSV original contra su SHA256 registrado:
0934296a198f159b98e486d9c2357ae72fd8dafcba7183ac148e823fe11e2a05.
La fila original 27 identifica Bolivia. Se agregaron sus conteos y se
reconstruyeron diferencias diarias con la biblioteca estándar, sin llamar
al repositorio ni a la función de métricas del producto.

Las 479 muestras van del 2020-02-06 al 2021-05-29. Entrenamiento contiene
383 fechas hasta el 2021-02-22; prueba, 96 fechas desde el 2021-02-23.
Los cinco folds usan 68, 131, 194, 257 y 320 muestras de ajuste y 63 de
validación cada uno. Se comprobó que sus fechas son anteriores al bloque
validado y que ninguna fecha de prueba participa en ajuste o validación.

Se reconstruyeron MAE, MSE, RMSE y R² de SVR lineal y las dos referencias
desde las diferencias originales y las predicciones persistidas. Coinciden
con report.json dentro de tolerancia relativa 1e-12 y absoluta 1e-10.
Las referencias coinciden exactamente con los retardos de uno y siete días.
Las medias y desviaciones poblacionales de los cinco RMSE guardados por
candidato coinciden con los resúmenes de validación. El mínimo entre los
siete candidatos aprendidos corresponde a svr_linear.

El visor HTML contiene el mismo JSON de la ejecución. El generador de
documentación lee ese registro para tablas y las salidas CSV para figuras;
no vuelve a entrenar ni recalcula métricas. El Word se renderizó para
comprobar tablas, figuras, caracteres, paginación y cifras contra el JSON.

## Interpretación y límites

SVR lineal obtuvo RMSE CV 247.0342, prueba 939.3127 y R² 0.1188.
La persistencia de un día tuvo un RMSE CV ligeramente menor (244.4203),
aunque en prueba su RMSE fue mayor (1236.1990). La persistencia semanal
obtuvo RMSE de prueba 1023.3459. Por ello elegir un candidato aprendido
no equivale a demostrar que siempre mejora una regla sencilla.

El 2021-05-25 la variación registrada fue 5696 y SVR lineal predijo
aproximadamente 796.10. No se atribuye ese salto a una causa concreta:
el archivo no contiene pruebas, incidencias de reporte ni contexto causal.
La RNA registró dos ConvergenceWarning por el límite de 2000 iteraciones.
No se cambiaron parámetros después de leer la prueba.

Las entradas de prueba incorporan antecedentes reales ya observados,
manteniendo el estimador fijo. No es pronóstico de todo el bloque desde
un único origen. La copia retrospectiva puede contener correcciones;
no se conocen las versiones efectivamente publicadas en cada fecha.
Esta prueba COVID queda históricamente observada tras la ejecución.

## Verificación ejecutable

Pasaron 48 pruebas, Ruff y mypy estricto sobre src, tests y scripts.
Trece pruebas corresponden a COVID. La batería cubre que cambios futuros
no alteren características anteriores, los escaladores aprendan con
sus folds y se conserven revisiones negativas. Los tests no vuelven a
ejecutar las 44 configuraciones del dataset histórico.
