# Registro de iteraciones

## 2026 09 29 Guía de estudio completa de los materiales disponibles

- Se añadió una guía Word y Markdown de las diapositivas 1.4 a 1.8, con fundamentos de apoyo, contraste localizado con Géron (2.ª edición), ejemplos resueltos y doce ejercicios con respuestas razonadas.
- Se explicaron también clasificación, logística, softmax y matriz de confusión, aunque el proyecto siga siendo regresión. Se corrigieron en la guía las definiciones FP/FN y el numerador de Fβ de las diapositivas.
- Se corrigió una lectura anterior del ejemplo fiscal: la rama Reembolso=Sí lleva correctamente a No; las ramas de la raíz no están invertidas. Se conserva el contexto histórico de la iteración 02 en sus notas.
- Se conservaron la guía específica de RNA y los informes. No se reentrenó ni se modificó el protocolo; las diapositivas 1.1–1.3 no estaban disponibles.
- Se verificaron cálculos pedagógicos, correspondencia con los JSON y la presentación del Word página por página.

## 2026 09 29 Iteración 03 e informe integrado

- Se confirmó el alcance exclusivo de vinos y se integraron avances 1 y 2 con una RNA, validación e ingreso manual en un único Word.
- Se fijó antes de entrenar un MLP 11→8→1 tanh con SGD; se conservaron CSV, semilla, partición y folds.
- La RNA obtuvo RMSE CV 0.6618 y RMSE de prueba histórica 0.6461. Se registraron seis avisos de falta de convergencia dentro de 1000 épocas; la prueba reutilizada se presenta como exploratoria.
- Se añadieron train, predict y lesson; la consulta usa parámetros guardados en JSON y no reentrena. El ejemplo manual procede de una fila histórica, no de una nueva medición del usuario.
- La guía relaciona diapositivas, ejercicio resuelto, tres ejercicios de práctica y depuración en PyCharm.
- Pasaron 35 pruebas, Ruff y mypy estricto; se verificaron derivadas, escalado por fold, equivalencia de inferencia y métricas del informe.
- Se conservaron informes anteriores y cambios locales del equipo. No se añadieron dependencias de entrenamiento.

## 2026 09 16 Iteraciones 01 y 02

- Se revisaron el informe original, su código, la base de vinos , distinguiendo lectura completa de apuntes y lectura focalizada del libro.
- Se detectó que el recorte IQR modificaba una tabla diferente de la exportada. El caso 01 conserva los valores originales, elimina duplicados exactos y valida el esquema.
- Se fijó la comparación de OLS, Ridge, CART, bosque aleatorio, SVR lineal y SVR RBF con referencia constante. Se separó selección por validación de la prueba final.
- Se inició una arquitectura por casos, tipada y con pruebas de integridad experimental, acuerdos para agentes y documentación de fuentes.
- Pasaron 21 pruebas científicas, Ruff y mypy estricto. La auditoría independiente reconstruyó métricas desde las predicciones y verificó todas las particiones.
- La selección eligió el bosque aleatorio: RMSE CV 0.6385 y de prueba 0.6364, frente a 0.8191 de la referencia en prueba.
- Se incorporaron dos Word, una vista HTML local y un registro de dependencias con uv. Los originales quedan como respaldo local.
- El usuario autorizó crear el repositorio y eligió expresamente su visibilidad pública: JARO-Hub/wine-quality-machine-learning.
