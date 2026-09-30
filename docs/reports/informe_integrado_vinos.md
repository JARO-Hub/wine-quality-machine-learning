# Calidad del vino tinto

Avances de preprocesamiento selección de modelos y red neuronal

Universidad Mayor de San Simón · Facultad de Ciencias y Tecnología · Departamento de Informática y Sistemas

Grupo 15 · Daniel Jose Reque Mendoza · David Oliver Chambi Villarroel · Jose Enrique Apaza Matias · Julian Angel Rodriguez Ortiz · Madahi Karen Condori Pérez

Ciencia de datos y machine learning · Cochabamba Bolivia · Septiembre de 2026

## I Introducción

Estudiamos cómo estimar la calidad de un vino tinto a partir de once mediciones fisicoquímicas. Este informe reúne la preparación de datos del avance 1, los seis regresores del avance 2 y una red neuronal artificial nueva. Incluye su validación, una prueba con entradas por consola y un ejercicio resuelto para comprender el aprendizaje. El proyecto se concentra en vinos.

El primer avance permite reconocer qué datos utilizamos; el segundo compara formas de relacionar entradas y calidad; la nueva etapa aprende esa relación con un perceptrón multicapa. El bosque mantiene la mejor media de RMSE de la comparación anterior, 0.6385. La RNA fija alcanza 0.6618 en validación y 0.6461 en la prueba histórica. No confirmamos convergencia de la RNA dentro de las 1000 épocas fijadas. Estas cifras describen el experimento y no prueban una superioridad general de una familia.

## II Objetivos

El objetivo general es construir y explicar un proceso reproducible que prepare mediciones del vino, compare modelos de regresión y permita consultar una red neuronal con valores introducidos por una persona.

Los objetivos específicos son verificar datos y duplicados; conservar la misma partición para las comparaciones; relacionar teoría, código y resultados; incorporar una RNA con propagación hacia delante y retropropagación; y distinguir una predicción individual de la evaluación estadística sobre observaciones con calidad conocida.

## III Marco teórico

En aprendizaje supervisado aprendemos de pares de entradas y respuestas conocidas. Usamos regresión porque buscamos una cantidad numérica. La respuesta `quality` es una puntuación ordinal: aproximamos como comparables las diferencias de un punto, sin afirmar que representen distancias sensoriales exactas. Géron distingue regresión y clasificación en la p. impresa 8, PDF 38.

Separar entrenamiento y prueba evita medir únicamente lo que el modelo ya aprendió. La validación cruzada compara decisiones dentro de entrenamiento; la prueba evalúa la decisión elegida. También el escalado debe aprender sus parámetros sin consultar validación o prueba. Esta separación se fundamenta en Géron, pp. impresas 69–74 y 79–80, PDF 99–104 y 109–110. Elegir 80/20, cinco folds y semilla 42 es una decisión del proyecto.

Una RNA combina sumas ponderadas y funciones de activación. El perceptrón multicapa describe su organización en capas. Feedforward describe el recorrido desde entradas hasta salida. Backpropagation obtiene derivadas del error respecto a pesos y sesgos; un optimizador utiliza esas derivadas para actualizarlos. Son componentes de una misma solución, no tres modelos que necesariamente deban programarse por separado. Las diapositivas RNA, pp. PDF 8 y 10–16, presentan ese recorrido; Géron lo explica en pp. impresas 289–293, PDF 319–323.

## IV Datos y avance de preprocesamiento

Wine Quality contiene propiedades fisicoquímicas y puntuaciones sensoriales de vinos portugueses. La fuente es Cortez y colaboradores y su distribución en UCI. Trabajamos con la copia local de vinos tintos, sin mezclar otras bases. La escala descrita por UCI va de 0 a 10; en nuestro archivo solo aparecen puntuaciones de 3 a 8.

| Verificación | Resultado |
| --- | ---: |
| Filas originales | 1599 |
| Entradas numéricas | 11 |
| Valores faltantes | 0 |
| Repeticiones exactas eliminadas | 240 |
| Filas conservadas | 1359 |
| Entrenamiento y prueba | 1087 y 272 |

Las entradas son acidez fija, acidez volátil, ácido cítrico, azúcar residual, cloruros, dióxido de azufre libre, dióxido de azufre total, densidad, pH, sulfatos y alcohol. La columna `quality` es exclusivamente la respuesta. El programa exige el mismo nombre y orden de columnas y conserva los identificadores originales.

Quitamos duplicados exactos conservando su primera aparición. Evitamos así que una copia idéntica se reparta entre entrenamiento y prueba. Esto no demuestra independencia entre botellas o cosechas porque el archivo no contiene esos identificadores. No imputamos: no hay datos faltantes.

El avance anterior diagnosticó valores extremos mediante el rango intercuartílico. Si Q1 y Q3 son los cuartiles primero y tercero, IQR es su diferencia. Se marcan valores menores que Q1 − 1.5 IQR o mayores que Q3 + 1.5 IQR. Esa regla identifica observaciones para revisar; no demuestra que sean errores. Además, el script anterior modificaba una tabla distinta de la que exportaba. Por ello conservamos los extremos y no afirmamos haber aplicado aquel recorte.

## V Avance de selección de modelos

### Notación y métricas

Una fila de once entradas se representa por x; su calidad conocida es y y la estimación del modelo es ŷ. Para varias observaciones, i identifica la fila y n el número de filas evaluadas. Definimos el residuo r como calidad real menos predicha. Sumar residuos permitiría cancelaciones, por lo que usamos cuadrados o valores absolutos:

$$r_i=y_i-\hat y_i,\qquad MSE=\frac{1}{n}\sum_i r_i^2,\qquad RMSE=\sqrt{MSE}.$$

MSE promedia errores cuadrados; RMSE vuelve a puntos de calidad y penaliza más los errores grandes. MAE es la magnitud absoluta media. Para R², la media de las respuestas del conjunto evaluado se representa por y con barra:

$$MAE=\frac{1}{n}\sum_i |r_i|,\qquad R^2=1-\frac{\sum_i r_i^2}{\sum_i(y_i-\bar y)^2}.$$

R² compara error con variación observada: puede ser negativo y no significa porcentaje de aciertos. Usamos RMSE como criterio principal y las otras tres métricas como descripciones complementarias. MSE y RMSE están matemáticamente relacionadas. La referencia predictiva devuelve siempre la media aprendida en entrenamiento; no utiliza la media de prueba para predecir.

### Dos modelos lineales

Para OLS, Ridge y SVR estandarizamos cada columna antes de ajustar. La media μ y la desviación s se calculan solo en el entrenamiento correspondiente. Llamamos u al valor transformado:

$$u_{ij}=\frac{x_{ij}-\mu_j}{s_j},\qquad \hat y_i=b+\sum_{j=1}^{11}w_j u_{ij}.$$

Aquí b es el intercepto y cada w es un coeficiente. OLS elige estos parámetros minimizando la suma de errores cuadrados. `LinearRegression` resuelve mínimos cuadrados; no implementamos descenso de gradiente para este candidato. Géron presenta el modelo y su solución numérica en pp. impresas 112–117, PDF 142–147.

Ridge conserva la misma predicción, pero penaliza coeficientes grandes. Con la convención de la biblioteca, α controla el peso de la penalización:

$$J_{Ridge}=\sum_i(y_i-\hat y_i)^2+\alpha\sum_j w_j^2.$$

No se penaliza b. Géron usa MSE más α del libro dividido por dos multiplicado por la suma de pesos cuadrados. Al multiplicar ese objetivo por n, su mínimo no cambia: el α equivalente en scikit-learn sería n α del libro / 2. Esta diferencia de convención impide copiar un valor de α entre ecuaciones sin revisar su escala (pp. impresas 135–137, PDF 165–167).

### Dos modelos con árboles

CART separa observaciones mediante umbrales de características. Cada hoja predice una constante c. Para entender cuál, minimizamos la suma de (y − c)² de las k filas de esa hoja. Su derivada es dos veces la suma de (c − y); al igualarla a cero obtenemos c igual a la media de esas k respuestas. Las divisiones buscan reducir el error cuadrático ponderado de las ramas. En regresión no usamos Gini ni entropía. La profundidad y el tamaño mínimo de hoja controlan cuánto puede fragmentar los datos (Géron, pp. impresas 183–184, PDF 213–214).

El bosque entrena B árboles con remuestreo y aleatoriedad de características. Si cada árbol entrega una predicción, la salida es su promedio. Árboles con errores parcialmente diferentes pueden compensarse; repetir árboles idénticos no aportaría esa ventaja. Usamos 160 árboles y validamos profundidad, mínimo de hoja y número de variables candidatas. Estos valores son decisiones propias, no una receta obligatoria del libro (Géron, pp. impresas 193–198, PDF 223–228).

### Dos regresores de vectores de soporte

SVR acepta una banda de tolerancia ε: no penaliza errores dentro de ella y cobra solo el exceso fuera. Para un residuo r, su pérdida es:

$$L_\varepsilon(r)=\max(0,|r|-\varepsilon).$$

Con calidad real 6, predicción 5.7 y ε = 0.1, la pérdida es 0.2. SVR lineal combina esta pérdida con un control de magnitud de los coeficientes; el parámetro C aumenta el peso de corregir excesos cuando crece. En SVR RBF se sustituye el producto lineal por una similitud entre entradas estandarizadas. Para dos entradas u y u′ y un parámetro positivo γ:

$$K(u,u')=\exp(-\gamma\lVert u-u'\rVert^2).$$

La distancia al cuadrado reúne las diferencias entre características; la exponencial transforma distancia en similitud. γ grande concentra la influencia en vecinos próximos. Esa similitud permite relaciones no lineales. Ambos candidatos usan `SVR`, cambiando el kernel; C y ε conservan su significado. Géron fundamenta kernels y SVR en pp. impresas 159–164, PDF 189–194.

### Protocolo y resultados del avance 2

Reservamos 20 % para prueba y usamos 80 % para entrenamiento, con semilla 42 y estratificación por calidad. Dentro de las 1087 filas de entrenamiento usamos cinco folds barajados con la misma semilla. Cada fold ajusta el modelo con cuatro partes y mide error en la quinta. `StandardScaler` está dentro de `Pipeline`; los árboles no requieren este escalado.

La búsqueda compara 43 configuraciones: una de OLS; cuatro α de Ridge; seis combinaciones de CART; ocho de bosque; seis de SVR lineal; y dieciocho de SVR RBF. Elegimos por menor media de RMSE de validación. La desviación entre folds expresa dispersión, no un intervalo de confianza. La configuración y los índices completos se conservan en el JSON del caso 02.

| Modelo | RMSE CV | Desviación CV |
| --- | ---: | ---: |
| OLS | 0.6659 | 0.0452 |
| Ridge | 0.6657 | 0.0457 |
| CART | 0.6862 | 0.0482 |
| Bosque aleatorio | 0.6385 | 0.0401 |
| SVR lineal | 0.6686 | 0.0509 |
| SVR RBF | 0.6464 | 0.0311 |

El bosque seleccionado tiene 160 árboles, profundidad sin límite explícito, mínimo de cinco filas por hoja y selección de la raíz cuadrada del número de características candidatas. Su ventaja validada frente a SVR RBF es pequeña y no demuestra significancia estadística. Solo el ganador y la referencia se evaluaron en prueba en esta etapa.

## VI Red neuronal y aprendizaje

### De las diapositivas a nuestra arquitectura

La diapositiva 6 suma entradas multiplicadas por pesos y resta un umbral. Nosotros usamos un sesgo aditivo igual al negativo de ese umbral. La diapositiva 7 muestra activaciones: necesitamos una función no lineal y diferenciable para propagar derivadas. Elegimos tanh para la capa oculta porque su derivada se expresa como 1 − h², siendo h su salida. Una función escalón no ofrece esa derivada útil. Géron explica activaciones en pp. impresas 291–292, PDF 321–322.

Proponemos una red 11 → 8 → 1: once mediciones, ocho neuronas ocultas y una salida numérica. Esta elección pequeña facilita el estudio; no se presenta como arquitectura óptima. Una salida lineal permite estimar calidad continua sin redondearla. Es coherente con la regresión de una sola respuesta descrita por Géron en pp. impresas 292–293, PDF 322–323. No usamos una salida por categoría porque no estamos clasificando vinos.

Para una fila estandarizada u, W1 contiene 11 × 8 pesos y b1 ocho sesgos; W2 contiene 8 × 1 pesos y b2 un sesgo. La multiplicación suma las contribuciones que llegan a cada neurona. Primero calculamos las ocho activaciones h; después su combinación de salida:

$$h=\tanh(uW_1+b_1),\qquad \hat y=hW_2+b_2.$$

Hay 105 parámetros aprendidos: 88 + 8 + 8 + 1. Las medias y escalas del preprocesamiento se aprenden por separado, exclusivamente con entrenamiento. Feedforward ejecuta estas operaciones hacia la salida y también se usa al predecir un vino nuevo.

### Qué calcula backpropagation

La diapositiva 11 compara salida obtenida y respuesta deseada mediante error cuadrático. Para explicar las derivadas, empezamos con una observación, sin regularización, y definimos E como la mitad del error cuadrático. El factor un medio elimina el dos que aparece al derivar:

$$E=\frac12(\hat y-y)^2,\qquad \frac{\partial E}{\partial\hat y}=\hat y-y.$$

No conocemos una respuesta correcta para cada neurona oculta. Esa dificultad aparece en la diapositiva 13. La solución de la diapositiva 14 es la regla de la cadena: medir cómo cambia el error con la salida y cómo cambia la salida con cada peso. Multiplicamos esas dependencias para retroceder hacia la capa oculta. No inventamos etiquetas para sus neuronas.

En una red didáctica de una neurona oculta, v conecta esa neurona con la salida; h es su activación. Si δ de salida es ŷ − y, el cambio del error respecto a la entrada de la neurona oculta es:

$$\delta_{salida}=\hat y-y,\qquad \delta_{oculta}=\delta_{salida}\,v\,(1-h^2).$$

Cada peso de entrada recibe como gradiente δ oculta multiplicado por su entrada. El gradiente de v es δ salida multiplicado por h. Los sesgos reciben sus correspondientes δ. Backpropagation calcula esas derivadas; descenso de gradiente utiliza una tasa η positiva para actualizar un parámetro cualquiera θ:

$$\theta_{nuevo}=\theta_{anterior}-\eta\frac{\partial E}{\partial\theta}.$$

La diapositiva 12 suma una corrección usando respuesta deseada menos predicha. Nosotros restamos el gradiente, que usa el signo opuesto: ambas formas coinciden para una salida lineal y este error. En capas con activación hay que incluir su derivada. El ejercicio del anexo muestra cada número antes de pasar a las once entradas del proyecto.

### Entrenamiento implementado

Usamos `MLPRegressor` con una capa de ocho neuronas tanh, SGD, tasa constante 0.01, lotes de 64, sin momentum ni Nesterov y semilla 42. Una época recorre todo entrenamiento mediante esos lotes. El máximo es 1000 épocas; la tolerancia es 0.00001 durante 50 épocas sin mejora suficiente; `early_stopping=False`. No se añade otra separación interna de validación.

La implementación incluye penalización L2 de pesos con α = 0.001. Para un lote de m observaciones, su pérdida escalar suma el medio error cuadrático medio y la penalización de los pesos de ambas capas:

$$J=\frac{1}{2m}\sum_i(\hat y_i-y_i)^2+\frac{\alpha}{2m}\left(\lVert W_1\rVert_F^2+\lVert W_2\rVert_F^2\right).$$

La norma al cuadrado suma los cuadrados de todas las entradas de cada matriz; no penalizamos sesgos. Así, a cada gradiente de pesos se añade α W / m. La derivación individual anterior se promedia por lote y recibe este término. Esta convención se comprobó con la implementación instalada de scikit-learn 1.9.1; no debe confundirse con α de Ridge. Las ocho neuronas y los hiperparámetros se fijaron antes de entrenar, sin buscar un resultado de prueba favorable.

## VII Validación y prueba de la RNA

Se conservan exactamente el CSV, las 1087 filas de entrenamiento, las 272 de prueba y los cinco folds del avance anterior. La RNA es un candidato fijo: realizamos cinco ajustes de validación y uno final sobre todo entrenamiento. Comprobamos que cada scaler aprende únicamente de las filas que le corresponden.

| Medición de la RNA | Resultado |
| --- | ---: |
| RMSE de entrenamiento final | 0.6130 |
| RMSE medio de validación | 0.6618 |
| Desviación del RMSE entre folds | 0.0310 |
| MAE medio de validación | 0.5155 |
| Épocas del ajuste final | 1000 |
| Avisos de convergencia en los seis ajustes | 6 |

Los seis ajustes alcanzaron el máximo de épocas sin confirmar convergencia. El modelo produce predicciones y métricas, pero no afirmamos que terminó de optimizar su pérdida. Conservamos esa limitación y sus avisos en el registro; no cambiamos parámetros después de ver prueba para mejorar sus cifras.

| Modelo en la prueba histórica | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Bosque del avance 2 | 0.4934 | 0.4050 | 0.6364 | 0.3963 |
| RNA fija del avance 3 | 0.4973 | 0.4174 | 0.6461 | 0.3777 |
| Referencia con media | 0.6921 | 0.6710 | 0.8191 | −0.0002 |

La RNA reduce el error frente a la referencia, pero esta ejecución no mejora la media de RMSE de validación del bosque. La tabla reúne el resultado histórico del bosque con la nueva ampliación: no se usó prueba para elegir un nuevo ganador entre siete modelos. Como el conjunto de prueba ya se había consultado, esta comparación es exploratoria y no una confirmación externa independiente. Una futura evaluación concluyente necesitará datos nuevos o un diseño de evaluación separado de las decisiones ya tomadas.

## VIII Prueba con valores ingresados por el usuario

Después de entrenar se guardan medias, escalas, pesos y sesgos en `model.json`. La acción `predict` carga esos números y solicita once mediciones con sus nombres y unidades. Acepta punto o coma decimal en el modo interactivo. Rechaza valores no finitos o negativos, densidad cero, pH fuera de 0–14 y alcohol fuera de 0–100 %. Son controles básicos para esta aplicación, no una certificación química completa.

Los valores fuera de los mínimos y máximos de entrenamiento generan advertencias. El programa conserva la medición: no recorta entradas ni salida. Si la salida lineal queda fuera de 0–10 también lo señala. Nunca vuelve a ajustar la escala con el vino introducido ni modifica los pesos durante la consulta.

Para comprobar el recorrido introdujimos por consola una fila conocida de la prueba histórica, identificador original 1152. Es una demostración reproducible del asistente con datos del archivo, no un vino nuevo aportado por el usuario:

| Medición | Valor |
| --- | ---: |
| Acidez fija en g/dm³ | 8.3 |
| Acidez volátil en g/dm³ | 0.6 |
| Ácido cítrico en g/dm³ | 0.25 |
| Azúcar residual en g/dm³ | 2.2 |
| Cloruros en g/dm³ | 0.118 |
| Dióxido de azufre libre en mg/dm³ | 9 |
| Dióxido de azufre total en mg/dm³ | 38 |
| Densidad en g/cm³ | 0.99616 |
| pH | 3.15 |
| Sulfatos en g/dm³ | 0.53 |
| Alcohol en porcentaje de volumen | 9.8 |

La RNA estima 5.3897; la calidad registrada es 5 y el error absoluto es 0.3897. Con la opción `--trace` se muestran también las once entradas estandarizadas y las ocho activaciones ocultas. Si no conocemos la calidad real, solo podemos informar la predicción. Conocerla permite calcular error individual, pero una sola observación no demuestra generalización ni permite un R² útil. Esta diferencia separa validación de entradas, evaluación estadística y demostración funcional.

## IX Organización y reproducción

Polars lee y valida las tablas; NumPy representa las matrices y scikit-learn ajusta modelos. La nueva RNA no necesita TensorFlow, GPU ni dependencias adicionales. El motor tabular de Polars usa Rust; no atribuimos a Rust todo el entrenamiento ni afirmamos ahorro de RAM sin una comparación medida.

Los casos 01, 02 y 03 separan preparación, selección y RNA. El caso 03 reutiliza el motor de particiones y evaluación del 02; los adaptadores construyen el modelo y guardan sus números. El ejercicio pedagógico está separado del entrenamiento real. Los contratos pequeños y la inyección de dependencias permiten sustituir componentes sin duplicar el procedimiento experimental. Se conservan tipos completos y una clase por archivo.

Desde la carpeta del proyecto, `uv sync --locked --extra dev` prepara el entorno. `uv run wine-neural train` ejecuta el experimento; `uv run wine-neural predict --trace` pide datos y muestra el recorrido; `uv run wine-neural lesson` ejecuta el ejercicio resuelto. Para depurar en PyCharm se usa el intérprete `.venv/bin/python`, el módulo `wine_quality.neural_cli` y uno de esos argumentos. El directorio de trabajo debe ser la raíz del repositorio.

Los resultados proceden de los archivos `report.json` de los casos 02 y 03; el ejemplo está en `manual_example.json`. Las ejecuciones registradas usaron Python 3.12.14, NumPy 2.5.3, Polars 1.44.2 y scikit-learn 1.9.1. El entorno local puede usar otra revisión compatible de Python; al reentrenar se registra la versión efectiva. No se confunden los 119592 bytes de X y 10872 de y con la RAM total del proceso.

Pasaron 35 pruebas, Ruff y mypy estricto. Se comprobaron particiones, escalado, métricas, coherencia de las ecuaciones anteriores, gradientes por diferencias finitas, persistencia de avisos y equivalencia entre predicción guardada y `Pipeline.predict`. La implementación queda acompañada por acuerdos de trabajo, una guía de estudio y el registro de decisiones.

## X Conclusiones

Unificamos la preparación y comparación anterior con una RNA explicable y una consulta por consola. La limpieza conserva mediciones originales y evita duplicados entre conjuntos; las transformaciones aprendidas permanecen dentro del entrenamiento correspondiente. No trasladamos al resultado procedimientos de imputación o recorte que el código no realiza.

La red 11–8–1 permite seguir el recorrido desde una medición hasta su calidad estimada y reconstruir cómo se calculan sus gradientes. Su desempeño mejora frente a predecir la media, pero no supera el RMSE validado del bosque y alcanzó el límite de épocas. Estas restricciones forman parte del resultado. Una entrada manual demuestra que el programa funciona; la evaluación de varios vinos con respuestas conocidas responde a una pregunta diferente sobre su error.

## Anexo Ejercicio de una actualización

Este ejemplo propio reduce la red a dos entradas, una neurona oculta tanh y una salida lineal. No son pesos aprendidos del vino. Las entradas son 1 y 0.5; sus pesos w1 y w2 son 0.2 y 0.4. El sesgo oculto b es 0.1; el peso de salida v es 0.6 y su sesgo c es 0.1. La respuesta deseada es 0.8 y la tasa η es 0.1. Usamos E igual a medio error cuadrático, sin L2.

Primero calculamos la suma oculta a, su activación h y la predicción:

$$a=1(0.2)+0.5(0.4)+0.1=0.5,\qquad h=\tanh(0.5)=0.462117.$$

$$\hat y=0.6h+0.1=0.377270,\qquad E=\frac12(0.377270-0.8)^2=0.089350.$$

La salida quedó por debajo del objetivo. Calculamos δ salida = −0.422730. Para retroceder multiplicamos por el peso v anterior y por la derivada de tanh: δ oculta = −0.422730 × 0.6 × (1 − 0.462117²) = −0.199473. Multiplicar cada δ por la entrada de su conexión produce los gradientes siguientes.

| Parámetro | Valor anterior | Gradiente | Valor nuevo |
| --- | ---: | ---: | ---: |
| w1 | 0.200000 | −0.199473 | 0.219947 |
| w2 | 0.400000 | −0.099736 | 0.409974 |
| b | 0.100000 | −0.199473 | 0.119947 |
| v | 0.600000 | −0.195351 | 0.619535 |
| c | 0.100000 | −0.422730 | 0.142273 |

Cada valor nuevo resulta de restar 0.1 veces su gradiente al valor anterior. Todos los gradientes se calculan con los parámetros anteriores; no se utiliza v actualizado para obtener δ oculta. Al repetir feedforward obtenemos 0.449980 y E = 0.061257. El error disminuyó en este paso, sin demostrar que cualquier tasa lo disminuiría siempre. El comando `lesson` permite verificar los números completos.

## Bibliografía

Géron, A. (2019). Hands On Machine Learning with Scikit Learn Keras and TensorFlow. Segunda edición. O’Reilly. Se consultaron los pasajes señalados del PDF local; sus páginas del archivo están 30 posiciones después de las impresas.

Rodríguez Bilbao, P. Material de la asignatura. Aprendizaje supervisado Deep Learning, archivo 1.8 Aprendizaje supervisado RNA2, 18 páginas. Se citan páginas PDF porque las láminas no muestran una paginación impresa diferente. Para los avances anteriores se consultaron sus materiales de preprocesamiento, validación, regresión y árboles.

Cortez, P., Cerdeira, A., Almeida, F., Matos, T. y Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016. Datos Wine Quality distribuidos por UCI con licencia CC BY 4.0: https://archive.ics.uci.edu/dataset/186/wine+quality.

Scikit-learn. MLPRegressor y Neural network models supervised. Documentación oficial consultada para la implementación; también se contrastó el código instalado de la versión 1.9.1. https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html y https://scikit-learn.org/stable/modules/neural_networks_supervised.html. Consulta de la RNA el 29 de septiembre de 2026.
