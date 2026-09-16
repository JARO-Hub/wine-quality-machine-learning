# Caso 02 Selección

Entrada: Dataset del caso01 y ExperimentConfig. El servicio recibe catálogo y motor por inyección. Salida: report.json con selección, auditoría de particiones y prueba del ganador frente a la media.

Métrica principal: media de RMSE en5 folds, semilla42. Prueba20% estratificada por quality. Pipelines aprenden escala dentro de cada entrenamiento. El catálogo define43 combinaciones entre6 candidatos.

No añadir mediciones de prueba de todos los candidatos para elegir posteriormente. Para03 conservar la estructura pero escribir primero la nueva pregunta y registrar que esta prueba ya se observó. Los contratos están en shared y evaluation_engine.py; los detalles sklearn, en adapters. El visor HTML consume resultados y no interviene en selección.
