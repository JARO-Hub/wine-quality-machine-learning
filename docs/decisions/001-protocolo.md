# Decisión 001 Comparación de seis regresores

Estado: adoptada para la primera iteración ejecutable.

## Pregunta

¿Qué relación aprendida entre once mediciones fisicoquímicas y la puntuación quality reduce el error de predicción en vinos no usados para ajustar el modelo?

Se continúa la regresión del trabajo previo. quality es ordinal; tratar diferencias de un punto como comparables es una aproximación operativa. No se afirma que la distancia sensorial sea exactamente igual entre categorías ni que los coeficientes sean efectos causales.

## Modelos

Regresión lineal ordinaria y Ridge; árbol de decisión regresor y bosque aleatorio regresor; SVR lineal y SVR con kernel RBF. Un bosque agrega muchos árboles y aleatoriza candidatos de variables para reducir su correlación. El SVR conserva la pérdida epsilon-insensible al cambiar el kernel. El predictor constante con la media es una referencia, adicional a los seis modelos.

## Datos y particiones

Quitar exclusivamente duplicados exactos conservando la primera fila. Conservar todos los predictores y no recortar extremos en esta iteración. El código anterior recortaba una tabla distinta de la exportada; por eso no se toma ese tratamiento como algo ya aplicado correctamente.

Reservar 20 % para prueba con semilla 42 y estratificación por quality. Este balance de categorías es una decisión nuestra, no una transformación del problema en clasificación. En el 80 % restante, KFold de 5 particiones barajadas con semilla 42. StandardScaler dentro de Pipeline para modelos lineales y SVR, sin escala para árboles. No normalizar quality.

La deduplicación global evita copias idénticas en lados distintos; no prueba independencia entre botellas ni entre cosechas porque no tenemos esos identificadores. Si se incorporan lotes o fechas, revisar validación por grupos o temporal.

## Selección y prueba

Elegir la menor media de RMSE por fold de validación cruzada. Mostrar desviación entre folds como dispersión descriptiva, no como intervalo de confianza. Reajustar el ganador con todo entrenamiento y medir MAE, MSE, RMSE y R² en prueba frente al predictor de media. No consultar pruebas de los otros cinco modelos para seleccionar o retocar.

La búsqueda usa entrenamiento y validación para seleccionar; su mejor score puede ser optimista. La prueba reservada evalúa la decisión completa una vez. Tras leerla, una iteración 03 deberá declarar la reutilización de ese conjunto o reservar nueva evidencia. No afirmar independencia para infinitas iteraciones sobre la misma prueba.

## Consigna y alcance

La práctica docente pide regresión y árboles, al menos cuatro parámetros de validación y matriz de confusión. La petición actual añade SVM. Ofrecemos cuatro métricas de regresión; una matriz de confusión exigiría una regla de clasificación explícita. Esa parte de la consigna queda pendiente de una tarea de clasificación, no se inventan clases ni se redondean predicciones para aparentar cumplimiento.

COVID queda como material independiente sin objetivo definido. No mezclar registros epidemiológicos con mediciones de vino.
