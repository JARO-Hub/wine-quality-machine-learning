# Evidencia de la RNA fija

- `report.json`: datos, huella, versiones, particiones, validación, métricas, predicciones y curva de pérdida. `models[0].fit_warnings` registra seis avisos de convergencia.
- `model.json`: parámetros numéricos de StandardScaler y de la red 11→8→1, nombres de entradas y procedencia. Sirve para inferencia sin volver a entrenar.
- `manual_example.json`: recorrido introducido por el asistente con la fila histórica 1152, calidad real 5 y predicción 5.3897. No es una observación nueva aportada por el usuario.

Se conservaron exactamente las filas y folds de `02_model_selection/report.json`. La prueba ya había sido observada; esta ampliación es exploratoria. La consola permite probar tus propios valores con `uv run wine-neural predict --trace`. El JSON del ejemplo no se modifica al consultar otros vinos salvo que se indique explícitamente esa ruta como salida.
