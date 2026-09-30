# Datos de vinos

`raw/winequality-red.csv` es una copia exacta del archivo aportado. Conserva comas como separador y nombres con espacios. No se descargó otro archivo para sustituirlo.

- 1599 filas, 11 entradas y `quality` como salida.
- 240 repeticiones exactas después de la primera aparición; quedan 1359 observaciones con la política adoptada.
- SHA256: `d6a0d9bd24806944818795f22500c46cb6424cbff517aacda36595d3ed9b2daa`.
- Procedencia bibliográfica: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., y Reis, J. (2009). *Wine Quality*. UCI Machine Learning Repository. [DOI 10.24432/C56S3T](https://doi.org/10.24432/C56S3T).
- Fuente y licencia declarada por UCI: [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

UCI documenta vinos tintos y blancos del norte de Portugal, variables fisicoquímicas y una puntuación sensorial de 0 a 10. En esta copia de vinos tintos sólo se observan valores de 3 a 8. La portada de UCI muestra un conteo de 4898 asociado a la variante blanca; no se usa ese número para describir el CSV local. No se comprobó igualdad byte por byte con el archivo remoto, cuya presentación usa otro separador.

La deduplicación define nuestra muestra analítica, pero la ausencia de identificadores impide demostrar que cada coincidencia sea un error de captura. Conservar la copia original permite reconsiderar esta elección.

`processed/` contiene salidas regenerables del caso 01. La carpeta COVID mencionada por el usuario sigue en su ubicación original y no interviene en este caso: no se ha definido todavía una pregunta ni un protocolo temporal para ella.
