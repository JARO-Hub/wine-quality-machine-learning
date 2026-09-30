# Evidencia temporal de COVID: Bolivia

Primera evaluación del protocolo 004, ejecutada el 30 de septiembre de 2026.
El CSV original y los resultados de vinos permanecen independientes.

- `preprocessing.json`: huella, país, fechas, filas originales y auditoría de preparación.
- `series.csv`: acumulado agregado y variación diaria; la primera diferencia queda vacía.
- `supervised.csv`: 18 entradas, respuesta, fecha objetivo e índice de fecha original.
- `report.json`: 44 configuraciones, cinco folds temporales, elección por RMSE,
  parámetros, avisos, versiones, tiempos y predicciones de prueba.
- `results.html`: visor autónomo construido desde el JSON.
- `series.png` y `evaluation.png`: figuras reconstruidas por el generador del informe.

SVR lineal fue elegido por RMSE CV 247.0342. La prueba (2021-02-23 a 2021-05-29)
dio RMSE 939.3127, MAE 546.0457, MSE 882308.3212 y R² 0.1188. Persistencia
de un día: RMSE 1236.1990; semanal: 1023.3459. La RNA emitió dos avisos
ConvergenceWarning por alcanzar el límite de iteraciones. No se cambiaron
configuraciones tras leer prueba.

La evaluación es secuencial a un día: parámetros fijos y antecedentes observados
actualizados diariamente. No es un pronóstico simultáneo de todo el periodo.
Este bloque de prueba ya fue observado; cualquier nueva decisión debe reconocerlo.
La consola impide sobrescribir una carpeta evaluada; usa otro `--output`.
