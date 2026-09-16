# Pruebas

Priorizar invariantes científicos: ninguna fila compartida entre entrenamiento y prueba, folds contenidos en entrenamiento, estadísticas de escalado aprendidas sólo con sus filas de ajuste, ecuaciones concordantes con la biblioteca y métricas recalculables desde predicciones.

Los tests deben usar ejemplos pequeños o el CSV público con procedencia documentada. No entrenar seis grillas completas en cada test, no congelar tiempos y no seleccionar una semilla por sus resultados. La integración completa se verifica con la CLI y report.json. Mantener las mismas reglas de tipado y una clase por archivo que en src.
