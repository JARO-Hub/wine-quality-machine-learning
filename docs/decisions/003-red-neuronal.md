# Decisión 003 Red neuronal e ingreso manual

Fecha: 29 de septiembre de 2026. Alcance confirmado: únicamente vinos.

## Pregunta y lectura

El nuevo requisito añade una RNA y una demostración con valores ingresados por una persona. Feedforward describe el recorrido hacia la salida; perceptrón multicapa describe capas y conexiones; backpropagation calcula los gradientes usados para aprender. No son tres modelos obligatorios distintos. Las diapositivas RNA, pp. 6–7 y 10–16, y Géron, impresas 289–293 / PDF 319–323, sustentan esta lectura.

La diapositiva 6 resta un umbral; en nuestro código sumamos un sesgo igual al negativo de ese umbral. La regla delta de la p. 12 omite la derivada de activación: es la forma para salida lineal y medio error cuadrático, no una regla general para todas las activaciones. La p. 14 incorpora la regla de la cadena. En las láminas Z significa respuesta deseada; aquí reservamos y para esa respuesta y U para entradas estandarizadas.

## Decisión antes de entrenar

Se añade un único candidato fijo StandardScaler → MLPRegressor: 11 entradas, 8 neuronas ocultas con tanh y una salida lineal. Hay 105 parámetros: 11×8 pesos + 8 sesgos + 8 pesos de salida + 1 sesgo. SGD con tasa constante 0.01, sin momentum ni Nesterov, lotes de 64, hasta 1000 épocas, alpha=0.001, tol=0.00001, n_iter_no_change=50, early_stopping=False y semilla 42. Esta red pequeña mantiene la explicación próxima a las láminas; no se elige por mirar su prueba. No requiere bibliotecas nuevas.

Se reutilizan sin cambios limpieza, 80/20 y cinco folds del caso 02. Cada scaler se ajusta dentro del fold. El único candidato se valida y reajusta en entrenamiento; solo él y la referencia se evalúan en prueba. La prueba ya fue vista en septiembre: este resultado es una ampliación exploratoria del mismo experimento, no una confirmación externa independiente. La comparación con las seis cifras históricas será descriptiva; no elegimos un nuevo ganador global mirando test.

La pérdida de MLPRegressor con salida escalar es SSE/(2m)+alpha·suma(W²)/(2m); los sesgos no se penalizan. En SGD se usa el tamaño del lote efectivo m. Backpropagation calcula derivadas; SGD resta tasa×gradiente. El error del ejercicio manual no tiene regularización y contiene una sola observación para poder calcular cada paso; esta diferencia debe quedar explícita.

## Uso y sencillez

La consola tendrá tres acciones: train, predict y lesson. train reutiliza el motor existente y guarda métricas, curva de pérdida y parámetros numéricos en JSON. predict solicita once mediciones con sus nombres, valida números finitos y dominio básico, conserva escala y pesos ya ajustados y entrega una calidad continua. Si se aporta una calidad real, añade residuo, error absoluto y error cuadrático. Una observación no permite estimar R² ni demostrar generalización. Los valores fuera del intervalo de entrenamiento generan una advertencia, sin recorte automático.

La persistencia usa JSON con arrays numéricos, no objetos ejecutables. Una predicción manual no ajusta el scaler ni la red. El recorrido de inferencia se reconstruye como U=(X-media)/escala, H=tanh(UW1+b1), salida=HW2+b2 y se contrasta con Pipeline.predict en una prueba.

lesson ejecuta un paso completo de una red didáctica 2→1→1, con gradientes comprobados mediante diferencias finitas. La guía de estudio contiene correspondencia con cada lámina y ejercicios; el informe único integra los avances con la RNA y limita el detalle de programación a un recorrido explicable.

## Informe recibido

El PDF de 20 páginas mezcla vinos y COVID; el usuario confirmó excluir COVID del nuevo alcance. El script adjunto solo reproduce carga, deduplicación y OLS; no verifica las seis implementaciones descritas ni el pipeline con imputación, capping y MinMaxScaler. Se conserva el trabajo ejecutable anterior como evidencia. No se trasladan afirmaciones sin respaldo, resultados de COVID ni instrucciones editoriales accidentales como «Puedes poner».

## Resultado y límite registrado

Los cinco folds y el reajuste alcanzaron 1000 épocas sin confirmar convergencia. Se conserva esta configuración y se registran seis ConvergenceWarning; no se ajustó la arquitectura después de observar prueba. RMSE CV 0.661768 y prueba histórica 0.646093. Una segunda ejecución idéntica verificó el registro estructurado de avisos; no cambió datos, particiones, parámetros ni predicciones. Es una repetición técnica de la misma evidencia, no un experimento independiente.
