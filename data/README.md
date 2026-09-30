# Datos de vinos

`raw/winequality-red.csv` es una copia exacta del archivo aportado. Conserva comas como separador y nombres con espacios. No se descargó otro archivo para sustituirlo.

- 1599 filas, 11 entradas y `quality` como salida.
- 240 repeticiones exactas después de la primera aparición; quedan 1359 observaciones con la política adoptada.
- SHA256: `d6a0d9bd24806944818795f22500c46cb6424cbff517aacda36595d3ed9b2daa`.
- Procedencia bibliográfica: Cortez, P., Cerdeira, A., Almeida, F., Matos, T., y Reis, J. (2009). *Wine Quality*. UCI Machine Learning Repository. [DOI 10.24432/C56S3T](https://doi.org/10.24432/C56S3T).
- Fuente y licencia declarada por UCI: [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

UCI documenta vinos tintos y blancos del norte de Portugal, variables fisicoquímicas y una puntuación sensorial de 0 a 10. En esta copia de vinos tintos sólo se observan valores de 3 a 8. La portada de UCI muestra un conteo de 4898 asociado a la variante blanca; no se usa ese número para describir el CSV local. No se comprobó igualdad byte por byte con el archivo remoto, cuya presentación usa otro separador.

La deduplicación define nuestra muestra analítica, pero la ausencia de identificadores impide demostrar que cada coincidencia sea un error de captura. Conservar la copia original permite reconsiderar esta elección.

Las salidas regenerables del caso 01 están en `outputs/01_preprocessing/`.

## COVID: copia histórica aportada

`raw/time_series_covid_19_confirmed.csv` contiene 276 filas geográficas,
193 etiquetas de país/región y 494 fechas consecutivas (2020-01-22 a 2021-05-29).
SHA256 de los bytes originales:
`0934296a198f159b98e486d9c2357ae72fd8dafcba7183ac148e823fe11e2a05`.
No se sustituyó ni modificó el archivo aportado.

El esquema coincide con la serie global de casos confirmados JHU CSSE. No se
comprobó igualdad byte por byte con un commit remoto ni se conoce la cadena
exacta de descarga del archivo renombrado. La atribución de contexto es
[JHU CSSE COVID-19 Data](https://github.com/CSSEGISandData/COVID-19).
Su README, apartado Terms of Use, declara CC BY 4.0 y solicita citar Dong, E.,
Du, H., y Gardner, L. (2020), *An interactive web-based dashboard to track
COVID-19 in real time*, doi:10.1016/S1473-3099(20)30120-1.

El caso 04 usa inicialmente Bolivia (fila original 27, índice desde cero),
agrega las filas del país y calcula diferencias diarias sin recorte. Sus
salidas están en `outputs/04_covid/`. Leer el protocolo 004 antes de ejecutar
otro país. Estos son registros históricos, no mediciones actuales ni fechas
de infección. La instantánea puede incluir revisiones retrospectivas.
