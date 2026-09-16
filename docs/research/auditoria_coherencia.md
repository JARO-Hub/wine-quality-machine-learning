# Auditoría independiente de coherencia

Revisión: 16-09-2026. Alcance: código de `src/wine_quality`, registro `outputs/02_model_selection/report.json` e informe `docs/reports/informe_vinos.md`, contrastados con `lectura_fuentes.md`. Revisión de lectura y comprobaciones independientes; no se modificó código de producto ni se volvió a seleccionar un modelo.

## Resultado

No se detectaron errores que invaliden la ejecución examinada: no hay fuga observable de información entre entrenamiento y prueba en el procedimiento, las fórmulas principales corresponden a los estimadores y los resultados numéricos del informe coinciden con el registro. Las precisiones de redacción indicadas al final no cambian resultados.

## Evidencia revisada

- **Datos y particiones.** Se verificó la huella SHA256 contra `data/raw/winequality-red.csv`, las 1599 filas originales, las 1359 primeras apariciones únicas y las 240 repeticiones eliminadas. Los 1087 identificadores de entrenamiento y los 272 de prueba no se solapan y cubren exactamente las filas conservadas. Tampoco existen vectores de predictores idénticos repartidos entre ambos conjuntos. Cada fold divide exclusivamente entrenamiento, sin solapamiento interno; cada observación de entrenamiento aparece una vez como validación.
- **Transformaciones y selección.** `sklearn_model_catalog.py` incorpora `StandardScaler` en los pipelines lineales y SVR. `GridSearchCV` recibe únicamente la matriz de entrenamiento y las particiones relativas a ella. `sklearn_evaluation_engine.py` elige por RMSE de validación antes de acceder a las características de prueba. Se registran métricas de prueba únicamente para ganador y referencia. La media del predictor constante procede de entrenamiento.
- **Objetivos matemáticos.** `LinearRegression` corresponde a mínimos cuadrados; `Ridge(solver="svd")`, a SSE más penalización alpha por suma de pesos cuadrados. La corrección de normalización respecto al libro está explícita. CART usa `squared_error` y el informe define media y MSE de nodo con sus divisores. El objetivo epsilon-insensible con norma de pesos está expresamente rotulado **SVR lineal**, por lo que no se está afirmando que el RBF tenga sólo once coeficientes originales. La expansión mediante vectores de soporte conecta correctamente ambos kernels.
- **Bosque.** El objeto construido en el entorno `.venv` tiene `bootstrap=True`, `max_samples=None`, 160 árboles y criterio cuadrático. La búsqueda compara `max_features=1.0` y `"sqrt"`; el ganador registrado usa `"sqrt"`. Por tanto, la afirmación de selección parcial aleatoria de variables sí describe al ganador. El candidato con 1.0 examina todas las variables y conserva aleatoriedad por bootstrap; la tabla del informe distingue ambas opciones.
- **Métricas.** Se recalcularon MAE, MSE, RMSE y R² de ganador y referencia directamente desde el CSV original y las predicciones registradas, sin llamar a la función de métricas del producto. Todos los valores coinciden dentro de 1e-12. Se verificaron también las medias y desviaciones poblacionales de los cinco RMSE por fold y que el mínimo selecciona al bosque. RMSE de prueba: 0.6363713853436372; referencia: 0.8191269109653824; reducción: 22.311014712770028 %. Diferencia de RMSE CV entre SVR RBF y bosque: 0.007842092500405684.
- **Descripción de extremos.** Se reprodujeron las 573 celdas y 405 filas marcadas por IQR sobre los once predictores originales. Esos conteos no incluyen `quality` y no se usan para recortar el entrenamiento actual.

## Precisiones acotadas sugeridas

1. En la explicación del bosque, especificar que el remuestreo se hace **con reemplazo**. Es el mecanismo efectivo del código y completa el puente desde bagging en Géron, pp. 193–194 / PDF 223–224.
2. Después de introducir RBF, aclarar: «La penalización corresponde a la norma del predictor en el espacio de características inducido por el kernel; no a once pesos sobre las columnas originales». La fórmula actual está bien delimitada al caso lineal, pero esta frase evita extrapolarla sin cambiar de espacio.

No se requieren cambios de hiperparámetros, nuevas búsquedas ni una nueva consulta a prueba como resultado de esta revisión.

## Límites

La comprobación demuestra coherencia con el CSV y los registros examinados, no independencia entre botellas o cosechas. No hay identificadores para comprobarla. La exploración previa del mismo archivo limita considerar la prueba completamente nueva, y el informe lo reconoce. La búsqueda puede favorecerse de su propia validación; su puntuación óptima no es una estimación insesgada. Tampoco se demuestra significancia estadística frente a SVR RBF ni ahorro de RAM: el informe evita ambas afirmaciones. La matriz de confusión de la consigna sigue pendiente de una tarea explícita de clasificación.

Se leyeron las pruebas científicas existentes, incluidas las que verifican estadísticos por fold, gradiente de Ridge, expansión SVR y promedio del bosque. No se repitió la suite ya informada como aprobada; esta auditoría agregó comprobaciones sobre el CSV y JSON reales, sin reentrenar candidatos.
