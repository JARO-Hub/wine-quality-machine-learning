# Preprocesamiento y comparación de modelos para calidad de vino tinto

Universidad Mayor de San Simón  
Facultad de Ciencias y Tecnología  
Departamento de Informática y Sistemas

Informe de avance del proyecto  
Grupo 15

Daniel Jose Reque Mendoza  
David Oliver Chambi Villarroel  
Jose Enrique Apaza Matias  
Julian Angel Rodriguez Ortiz  
Madahi Karen Condori Pérez

Materia Ciencia de datos y machine learning  
Cochabamba Bolivia  
Septiembre de 2026

## I Introducción

Este proyecto estudia si once mediciones fisicoquímicas permiten estimar la puntuación de calidad de un vino tinto. Continuamos el preprocesamiento realizado por el grupo y organizamos una comparación de seis modelos: regresión lineal ordinaria, Ridge, árbol de decisión, bosque aleatorio y dos regresores de vectores de soporte. El propósito es comprender qué aprende cada modelo, cómo se implementa esa idea y con qué evidencia podemos preferirlo.

El conjunto entregado contiene 1599 observaciones, once características y la respuesta `quality`. Confirmamos cero valores faltantes y 240 filas repetidas; al conservar una aparición de cada fila quedan 1359 observaciones. La revisión también detectó una discrepancia: el código anterior recortaba extremos en una tabla, pero guardaba otra copia sin ese recorte. En esta iteración conservamos los extremos como mediciones potencialmente válidas y usamos el rango intercuartílico únicamente para describirlos.

La comparación conserva una misma tarea y un mismo procedimiento de evaluación. El bosque obtuvo el menor RMSE de validación entre los candidatos probados y redujo el RMSE de prueba un 22.31 % respecto a predecir siempre la media de entrenamiento. Las explicaciones de Géron sustentan los conceptos; la deduplicación, las semillas, los candidatos y el protocolo concreto son decisiones de nuestro proyecto. Esta distinción evita presentar nuestras elecciones como instrucciones universales del libro.

## II Objetivos

### Objetivo general

Preparar los datos de vino tinto y comparar seis regresores mediante un procedimiento reproducible que permita explicar la relación entre su fundamento matemático, su implementación y su error de predicción.

### Objetivos específicos

1. Verificar estructura, valores faltantes, duplicados y extremos, manteniendo trazabilidad del archivo original.
2. Definir la respuesta, los predictores y las métricas antes de entrenar.
3. Explicar y construir dos candidatos de cada familia solicitada, conservando la misma tarea de regresión.
4. Seleccionar con validación cruzada sobre entrenamiento y evaluar el candidato elegido frente a una referencia constante.
5. Organizar código, resultados y documentación para continuar las siguientes iteraciones en equipo.

## III Marco teórico

En aprendizaje supervisado disponemos de ejemplos que relacionan entradas con una respuesta conocida. Una regresión estima una cantidad numérica; Géron distingue esta tarea de la clasificación, que asigna categorías (2019, p. 8; PDF p. 38). En nuestro caso, las entradas son propiedades fisicoquímicas y la respuesta es `quality`. Aunque sus registros son enteros ordenados, adoptamos una aproximación de regresión: tratamos una diferencia de un punto como comparable entre posiciones de la escala. Esto no demuestra que las diferencias sensoriales sean exactamente iguales.

El preprocesamiento debe conservar el significado de los datos y separar la detección de un problema de la decisión de corregirlo. Una fila idéntica a otra es una repetición verificable; un valor alejado estadísticamente no es, por ese solo hecho, una medición equivocada. Las transformaciones que calculan parámetros, como el escalado, deben aprenderlos de entrenamiento. Géron desarrolla la separación temprana de prueba y el ajuste del escalado con datos de entrenamiento (pp. 30–32 y 69–70; PDF pp. 60–62 y 99–100).

Evaluar sobre los mismos ejemplos usados para ajustar puede favorecer modelos que memoricen peculiaridades. La validación cruzada divide entrenamiento en partes y alterna cuál se reserva para validar; sirve para comparar decisiones sin consultar repetidamente la prueba (Géron, pp. 73–74; PDF pp. 103–104). El error final debe medirse después de seleccionar y ajustar el modelo (pp. 79–80; PDF pp. 109–110).

Los seis candidatos representan hipótesis distintas: una relación lineal, una relación lineal regularizada, particiones mediante árboles, un promedio de árboles y dos funciones con una tolerancia explícita al error. Las derivaciones y decisiones de cómputo se desarrollan en Ingeniería para mantener este marco teórico dentro del máximo de una página solicitado en la práctica docente.

## IV Marco referencial

Trabajamos exclusivamente con el archivo `winequality-red.csv` proporcionado. Sus columnas predictoras son `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates` y `alcohol`. No usamos la carpeta COVID: corresponde a otro conjunto sin objetivo definido para este experimento.

La fuente pública Wine Quality describe mediciones fisicoquímicas y una evaluación sensorial de vinos portugueses; distingue estos predictores de la calidad asignada. El archivo local es una copia separada por comas que debe identificarse por su huella, sin atribuirle modificaciones no comprobadas. Su rango observado de `quality` es 3–8; la escala completa descrita por la fuente es 0–10 (Cortez et al., 2009; [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)).

| Comprobación local | Resultado |
| --- | ---: |
| Observaciones originales | 1599 |
| Predictores | 11 |
| Valores faltantes | 0 |
| Repeticiones exactas después de la primera aparición | 240 |
| Observaciones conservadas | 1359 |

| Puntuación | Filas originales | Filas conservadas |
| --- | ---: | ---: |
| 3 | 10 | 10 |
| 4 | 53 | 53 |
| 5 | 681 | 577 |
| 6 | 638 | 535 |
| 7 | 199 | 167 |
| 8 | 18 | 17 |

Las puntuaciones centrales predominan. Por ello no basta con observar unos pocos aciertos ni con presentar un único ejemplo favorable. Además, el archivo carece de identificadores de botella, lote o cosecha: podemos evitar copias exactas entre particiones, pero no demostrar independencia entre procedencias.

Se revisaron siete fuentes PDF: la consigna y los materiales de preprocesamiento, validación, regresión, árboles y redes neuronales de Patricia Rodríguez Bilbao, además del libro de Géron. El material de árboles incluye las láminas de SVM; no es un archivo separado. Los seis documentos docentes se leyeron completos; del libro de 851 páginas se consultaron los pasajes pertinentes de los capítulos 1, 2, 4, 5, 6 y 7. La síntesis de lectura acompaña este informe. Redes neuronales se leyó para entender el curso, pero queda fuera de los seis candidatos.

## V Ingeniería

### Preparación coherente con la evidencia

Conservamos el CSV original y registramos su huella SHA256. Eliminamos exclusivamente duplicados exactos, conservando su primera aparición. Es una decisión operativa para este proyecto: no afirmamos que todas las repeticiones sean errores de laboratorio. La ausencia de nulos permite evitar imputaciones innecesarias. Las once entradas son numéricas, por lo que tampoco se requiere crear indicadores de categorías.

Para describir extremos usamos los cuartiles primero y tercero, `Q1` y `Q3`. Su diferencia mide la amplitud de la mitad central de las observaciones:

$$IQR=Q3-Q1,\qquad L=Q1-1.5IQR,\qquad U=Q3+1.5IQR.$$

Una celda fuera de `[L,U]` se marca para revisión. Sobre las 1599 filas originales aparecen 573 celdas marcadas, distribuidas en 405 filas; una fila puede aportar varias celdas. Esto reproduce el conteo del avance anterior, sin confundir ocurrencias con observaciones distintas. No recortamos esos valores: la detección no aportó evidencia suficiente de error. Si una futura iteración estudia recorte, los límites deberán ajustarse dentro de cada entrenamiento y compararse como otra decisión experimental.

### De observaciones a una función de predicción

Representamos las entradas de una observación mediante `xᵢ`, un vector de once componentes, y su calidad registrada mediante `yᵢ`. Al apilar las entradas obtenemos `X`, con `n` filas y once columnas; `y` contiene las `n` respuestas. La función aprendida `f` produce `ŷᵢ=f(xᵢ)`. Definimos el residuo como `rᵢ=yᵢ−ŷᵢ`: un residuo positivo significa que el modelo predijo por debajo de la puntuación registrada.

Esta notación conecta teoría e implementación: `quality` forma `y` y nunca debe entrar en `X`. No redondeamos las predicciones antes de evaluar. Una estimación de 5.8 expresa una salida numérica del modelo, no una nueva etiqueta sensorial.

Para los candidatos lineales y SVR estandarizamos cada predictor: `zᵢⱼ=(xᵢⱼ−μⱼ)/sⱼ`, donde `μⱼ` y `sⱼ` son la media y desviación calculadas en el entrenamiento correspondiente. La matriz `Z` reúne esos valores. Cambiamos su escala, no el significado de `y`; el ajuste de esta transformación se repetirá dentro de cada partición de validación.

### Del residuo a las métricas

Sumar residuos permitiría que errores positivos y negativos se cancelaran. Para medir discrepancia sin esa cancelación elevamos cada residuo al cuadrado y sumamos:

$$SSE=\sum_{i=1}^{n}r_i^2,\qquad MSE=\frac{SSE}{n},\qquad RMSE=\sqrt{MSE}.$$

SSE acumula error cuadrático; MSE lo expresa por observación. Si `n` es fijo, dividir por `n` no cambia qué predicciones minimizan la suma. La raíz es creciente sobre números no negativos, por lo que tampoco cambia ese orden; RMSE recupera la unidad de la respuesta: puntos de calidad. Los errores grandes siguen pesando más que los pequeños. Géron presenta MSE para entrenamiento lineal y RMSE como medida de rendimiento (pp. 40–42 y 113; PDF pp. 70–72 y 143).

Usamos RMSE como criterio principal y acompañamos el resultado con:

$$MAE=\frac1n\sum_i|r_i|,\qquad R^2=1-\frac{\sum_i r_i^2}{\sum_i(y_i-\bar y)^2}.$$

MAE informa la magnitud absoluta media del error. R² compara el error cuadrático con la variación respecto de la media del conjunto evaluado; puede ser negativo y no es porcentaje de aciertos. Para una referencia predictiva reproducible usamos además la media aprendida únicamente de entrenamiento. Esa referencia es distinta del denominador descriptivo de R². Las cuatro métricas mantienen la tarea de regresión; MSE y RMSE son expresiones relacionadas, no evidencias independientes. El requisito docente de una matriz de confusión queda pendiente de una tarea de clasificación con categorías y una regla explícita; no se inventan clases para aparentar cumplimiento.

### Regresión lineal ordinaria

Nuestro primer candidato representa la predicción como una suma de contribuciones:

$$\hat y_i=b+\sum_{j=1}^{11}w_jz_{ij}.$$

`b` es el intercepto y `wⱼ` el coeficiente de la característica `j`. Ajustarlos significa elegir valores que minimicen SSE. Un coeficiente describe una relación condicionada por los demás predictores; no demuestra que modificar químicamente una variable produzca ese cambio de calidad. La motivación y el ajuste por mínimos cuadrados se desarrollan en Géron, pp. 112–117; PDF pp. 142–147.

Para escribir el intercepto en la misma operación, añadimos a `Z` una columna de unos y llamamos `Z̃` a esa matriz; `θ` reúne intercepto y coeficientes. Una solución de mínimos cuadrados puede expresarse como `θ=Z̃⁺y`, donde `Z̃⁺` es la pseudoinversa. No necesitamos invertir directamente `Z̃ᵀZ̃`: esa inversa puede no existir si hay dependencia entre columnas. Géron explica la pseudoinversa y la descomposición en valores singulares (pp. 116–117; PDF pp. 146–147). En código delegamos la solución numérica en `LinearRegression`; no programamos una inversión manual.

### Ridge como control de coeficientes

El segundo candidato conserva la misma forma de predicción, pero añade un costo por coeficientes grandes. Para corresponder exactamente con la convención de `sklearn.linear_model.Ridge`, escribimos:

$$\min_{b,w}\;\sum_i(y_i-b-z_i^Tw)^2+\alpha\sum_jw_j^2.$$

`α` controla cuánto pesa la penalización respecto al ajuste; el intercepto no se penaliza. La penalización puede estabilizar coeficientes cuando las características aportan información relacionada, a cambio de restringir el ajuste. Se selecciona `α` con validación, sin declarar de antemano que un valor mayor siempre es mejor.

El libro presenta `MSE+(α_libro/2)‖w‖²` (Géron, pp. 135–137; PDF pp. 165–167). Multiplicar ese objetivo por `n` conserva su mínimo y produce `SSE+(nα_libro/2)‖w‖²`; por tanto, un mismo problema tiene `α_sklearn=nα_libro/2`. No copiamos valores numéricos de penalización entre convenciones como si fueran idénticos. La ecuación anterior coincide con la documentación oficial de Ridge.

La estandarización definida antes evita que las distintas unidades determinen por sí solas qué coeficiente resulta más costoso. Ridge reduce magnitudes de pesos; no implica eliminar automáticamente características ni volver sus coeficientes exactamente cero.

### Árbol de decisión para regresión

Un árbol divide el espacio mediante preguntas como «¿alcohol es menor o igual que un umbral?». Cada observación sigue esas decisiones hasta una hoja. Si una hoja contiene respuestas `y₁,…,yₖ`, buscamos una constante `c` que minimice `Σ(yᵢ−c)²`. La derivada es `2Σ(c−yᵢ)`; igualarla a cero da `c=Σyᵢ/k`. Por eso la predicción de una hoja es la media de sus respuestas, no una categoría mayoritaria.

Para elegir una división comparamos el error de sus dos ramas. Si el nodo tiene `n` observaciones y deja `n_L` a la izquierda y `n_R` a la derecha, usamos:

$$J=\frac{n_L}{n}MSE_L+\frac{n_R}{n}MSE_R,\qquad MSE_A=\frac1{n_A}\sum_{i\in A}(y_i-\bar y_A)^2.$$

Los pesos conservan la contribución de cada observación; no dan a una hoja diminuta la misma importancia que a una grande. CART busca una característica y un umbral que reduzcan este costo. Géron describe la regresión con árboles en pp. 183–184, PDF pp. 213–214. En esta edición, la ecuación 6-4 imprime la expresión de `MSE_node` sin su divisor; explicitamos la media para evitar mezclar suma y promedio.

Un árbol muy profundo puede adaptar hojas a peculiaridades del entrenamiento. Comparamos límites de profundidad y mínimos de observaciones por hoja con `DecisionTreeRegressor`. No requiere estandarizar para comparar umbrales por característica; sí requiere controlar su complejidad y evaluar generalización.

### Bosque aleatorio como promedio de árboles

Un bosque entrena varios árboles y promedia sus predicciones:

$$\hat y(x)=\frac1B\sum_{b=1}^{B}\hat y_b(x).$$

`B` es el número de árboles. El remuestreo de observaciones con reemplazo y la selección aleatoria de características candidatas en las divisiones hacen que los árboles no sean copias idénticas. Si todos cometieran exactamente el mismo error, promediarlos no lo corregiría. Si sus errores varían parcialmente, el promedio puede compensar parte de esa variación; por eso importa reducir la correlación entre árboles, además de aumentar su número (Géron, pp. 193–194 y 197–198; PDF pp. 223–224 y 227–228).

Como ejemplo pedagógico propio, tres árboles que predicen 5, 6 y 7 producen una media de 6. Si el valor real fuera 6, sus discrepancias se compensarían; tres predicciones iguales a 5 conservarían ese error. El ejemplo explica la utilidad de diversidad, sin garantizar compensación en todos los vinos. Implementamos `RandomForestRegressor` y validamos sus controles de complejidad. Más árboles implican más cómputo y no garantizan menor sesgo.

### Dos regresores de vectores de soporte

SVR introduce una banda de tolerancia `ε` alrededor de la predicción. Dentro de ella no cobra pérdida; fuera, cobra el exceso:

$$L_\varepsilon(r)=\max(0,|r|-\varepsilon).$$

Si observamos calidad 6 y predecimos 5.7, el error absoluto es 0.3. Con `ε=0.1`, la pérdida es `0.3−0.1=0.2`, no 0.3. La tolerancia es una elección del modelo y no prueba que un error de esa magnitud sea sensorialmente irrelevante. Géron explica la banda de SVR en pp. 162–164, PDF pp. 192–194; la documentación oficial de scikit-learn precisa su objetivo.

Para SVR lineal, la predicción es `b+wᵀz` sobre las características estandarizadas y el objetivo es:

$$\min_{b,w}\;\frac12\|w\|^2+C\sum_iL_\varepsilon(y_i-b-w^Tz_i).$$

El primer término controla la magnitud de los coeficientes y el segundo penaliza errores fuera de la banda. `C` grande da más peso a corregir esos excesos; no tiene el mismo papel ni la misma escala que `α` de Ridge. Comparamos `C` y `ε` utilizando `SVR(kernel="linear")`, manteniendo la pérdida ε-insensible.

El otro candidato mantiene esa lógica y permite relaciones no lineales mediante el kernel gaussiano:

$$K(z,z')=\exp(-\gamma\|z-z'\|^2).$$

El kernel convierte distancia en similitud. `γ` grande concentra la influencia en observaciones cercanas; `γ` pequeño la extiende. Como la distancia usa todas las componentes, el escalado resulta esencial para que una unidad grande no domine sin justificación. `SVR(kernel="rbf")` construye su función a partir de estas similitudes y observaciones de soporte, sin crear manualmente todas las características de un espacio expandido (Géron, pp. 159–161 y 171; PDF pp. 189–191 y 201). En RBF, la penalización corresponde a la norma del predictor en el espacio de características inducido por el kernel; ya no se limita a once pesos sobre las columnas originales. `C`, `ε` y `γ` se eligen conjuntamente mediante validación.

La predicción puede escribirse `f(z)=b+Σᵢ(aᵢ−aᵢ*)K(zᵢ,z)`. Los coeficientes `aᵢ` y `aᵢ*` se aprenden al resolver el objetivo con tolerancia descrito antes; muchas diferencias son cero y las observaciones con contribución no nula son vectores de soporte. Si el kernel es lineal, `K(zᵢ,z)=zᵢᵀz` y la suma se agrupa como `wᵀz+b`. Con RBF esa pendiente fija sobre las variables originales no basta. Esta identidad conecta ambos candidatos sin desarrollar innecesariamente la optimización dual.

### Protocolo común de selección

Reservamos 20 % de las 1359 observaciones para prueba y usamos 80 % para entrenamiento: 272 y 1087 filas. La semilla es 42 y estratificamos por `quality` para conservar aproximadamente sus proporciones; es una decisión de muestreo, no una conversión a clasificación. Dentro de entrenamiento aplicamos cinco particiones `KFold`, barajadas con semilla 42.

Cada candidato utiliza exactamente los mismos conjuntos. Para OLS, Ridge y SVR, `StandardScaler` queda dentro de `Pipeline`; así, cada ajuste calcula sus parámetros únicamente en la parte de entrenamiento del fold. Los árboles reciben las características sin escalado. El predictor constante con la media es una referencia adicional, no uno de los seis candidatos.

Seleccionamos la menor media de RMSE entre folds y mostramos su desviación, calculada con divisor cinco, como dispersión descriptiva, no como intervalo de confianza. Esa media no es idéntica al RMSE calculado al juntar todos los residuos. Reajustamos el ganador con todo entrenamiento y evaluamos únicamente ese modelo y la referencia sobre prueba. Los otros resultados de prueba no se usan para buscar un ganador más favorable.

El script anterior ya inspeccionó una prueba sin estratificación con semilla 42. La partición actual cambia, pero usa observaciones del mismo archivo: no constituye una prueba externa nueva. Además, una vez consultado su resultado, posteriores decisiones pueden adaptarse a él. La iteración siguiente deberá declarar su reutilización o incorporar evidencia adicional.

### Teoría del cómputo y organización

Polars carga, valida y manipula las tablas; su motor está desarrollado en Rust, según su [guía oficial](https://docs.pola.rs/). Convertimos las once columnas y la respuesta a arreglos NumPy para entrenar con scikit-learn. NumPy y scikit-learn no se presentan como bibliotecas íntegramente escritas en Rust. Esta separación permite usar una herramienta tabular eficiente sin cambiar algoritmos de aprendizaje ampliamente documentados.

El archivo tiene aproximadamente 99 KiB, por lo que no existe una necesidad demostrada de optimización de RAM. Registramos el tamaño de los arreglos mediante `nbytes`; eso mide sus buffers, no el consumo total del proceso ni prueba ahorro frente a otra biblioteca. Los tiempos observados tampoco deben confundirse con una comparación controlada de rendimiento.

La organización por casos de uso separa preparación y selección. El contrato de `src/wine_quality/cases/case_01_preprocessing/service.py` consiste en validar el archivo y producir datos y conteos trazables; el de `src/wine_quality/cases/case_02_model_selection/service.py` consiste en construir candidatos, seleccionar y emitir resultados. Las interfaces y adaptadores compartidos viven en `src/wine_quality/shared/`. Una etapa 03 podrá reutilizar la organización, manteniendo decisiones propias explícitas.

Aplicamos SOLID mediante responsabilidades verificables: una unidad carga datos; otra define particiones; otra construye modelos; otra evalúa; otra presenta. El catálogo permite añadir candidatos sin cambiar el procedimiento de evaluación. Los componentes sustituidos deben cumplir el mismo contrato de entradas y salidas; las interfaces se mantienen pequeñas y se suministran dependencias en lugar de ocultarlas en variables globales. Se usa una clase por archivo cuando existe una responsabilidad que justifique esa clase, con atributos internos protegidos por su interfaz pública y anotaciones de tipos completas.

La consola muestra lo esencial y los resultados alimentan una vista HTML sencilla, que no reimplementa entrenamiento. `AGENTS.md`, documentación de cada caso y registro de cambios fijan acuerdos para el equipo. Pasaron 21 pruebas y las revisiones de Ruff y mypy estricto. Las pruebas comprueban separación de datos, escalado por entrenamiento, métricas desde predicciones y correspondencia matemática: gradiente de Ridge, medias de hojas CART, promedio del bosque y expansión por kernel de SVR. Así se revisa lo que hace el código frente a las ecuaciones propuestas.

### Resultados y contraste con el objetivo

La ejecución registrada en `outputs/02_model_selection/report.json` usó Python 3.12.14, NumPy 2.5.3, Polars 1.44.2 y scikit-learn 1.9.1. La búsqueda fue acotada, con los siguientes valores; no representa una optimización exhaustiva de cada familia. Cada combinación se comparó en las mismas cinco particiones.

| Candidato | Valores explorados | Combinaciones |
| --- | --- | ---: |
| OLS | Sin hiperparámetros de búsqueda | 1 |
| Ridge | α: 0.1, 1, 10, 100 | 4 |
| CART | Profundidad: 3, 5, sin límite; mínimo de hoja: 5, 15 | 6 |
| Bosque | 160 árboles; profundidad: 8, sin límite; mínimo de hoja: 2, 5; variables candidatas: todas, raíz cuadrada | 8 |
| SVR lineal | C: 0.1, 1, 10; ε: 0.1, 0.2 | 6 |
| SVR RBF | C: 1, 10, 100; ε: 0.1, 0.2; γ: `scale`, 0.01, 0.1 | 18 |

`scale` calcula γ a partir del número de predictores y la varianza de las entradas de entrenamiento. Se fijó un trabajador de ejecución y una caché de 64 MB por SVR. Las 43 combinaciones de candidatos implican 215 ajustes de validación, además de los reajustes y la referencia.

| Candidato con su mejor configuración | RMSE entrenamiento | RMSE CV medio | Desviación CV |
| --- | ---: | ---: | ---: |
| OLS | 0.6595 | 0.6659 | 0.0452 |
| Ridge | 0.6595 | 0.6657 | 0.0457 |
| CART | 0.6614 | 0.6862 | 0.0482 |
| Bosque aleatorio | 0.4667 | 0.6385 | 0.0401 |
| SVR lineal | 0.6626 | 0.6686 | 0.0509 |
| SVR RBF | 0.5565 | 0.6464 | 0.0311 |

El bosque elegido tiene 160 árboles, profundidad sin límite explícito, mínimo de cinco observaciones por hoja y `max_features="sqrt"`; en esta configuración sí se restringen aleatoriamente las variables candidatas. Ridge eligió α=10; CART, profundidad 3 y hoja mínima 5; SVR lineal, C=0.1 y ε=0.2; SVR RBF, C=1, ε=0.2 y γ=`scale`.

El bosque superó a SVR RBF por aproximadamente 0.0078 puntos de RMSE medio. Es una diferencia pequeña: esta comparación no establece significancia estadística ni superioridad universal. Además, su error de entrenamiento, 0.4667, es menor que el validado, 0.6385; esa brecha advierte que parte de su ajuste no se mantiene fuera de las observaciones entrenadas.

| Evaluación sobre las 272 filas de prueba | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Bosque seleccionado | 0.4934 | 0.4050 | 0.6364 | 0.3963 |
| Referencia con media de entrenamiento | 0.6921 | 0.6710 | 0.8191 | −0.0002 |

La referencia predice 5.6256 para todos los vinos. El bosque reduce su RMSE un 22.31 %, calculado como `100×(1−0.636371/0.819127)`. Su MAE equivale a unas 0.49 unidades de puntuación en esta partición; no significa que todos los errores sean de media unidad. Sólo se evaluaron en prueba el ganador y esta referencia.

Los buffers de `X` e `y` ocupan 119592 y 10872 bytes respectivamente; son tamaños de arreglos, no RAM total. El registro conserva también parámetros, tiempos, identificadores de filas y predicciones para revisar los resultados. La conclusión se limita a este archivo, este protocolo y esta búsqueda.

## VI Conclusiones

La revisión conserva los resultados comprobables del primer avance y corrige su principal divergencia: el archivo empleado por el modelo anterior no contenía el recorte que describía el informe. La iteración actual adopta una política explícita de deduplicación y conservación de extremos, con separación entre observaciones originales, transformaciones aprendidas y evaluación.

Los seis candidatos responden a la misma pregunta, pero usan mecanismos distintos. Sus ecuaciones permiten reconocer qué calcula cada implementación: minimizar cuadrados, penalizar coeficientes, dividir por error, promediar árboles o controlar excesos fuera de una banda. Esta conexión permite revisar el código sin introducir modelos o fórmulas que no tengan una función explicada.

El bosque fue el candidato seleccionado por validación y alcanzó RMSE de 0.6364 en prueba, frente a 0.8191 de la referencia. La ventaja pequeña sobre SVR RBF durante selección y la reutilización de un archivo ya explorado limitan la fuerza de la comparación. Ni la complejidad del algoritmo ni una biblioteca escrita en Rust demuestran por sí mismas una mejora. El procedimiento queda documentado para repetirlo y modificarlo sin perder el vínculo entre teoría, decisiones y evidencia.

## Bibliografía

Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. Segunda edición. O’Reilly. Se consultó el PDF local; las referencias indican página impresa y página del archivo, con un desfase de 30 páginas en los pasajes citados.

Cortez, P., Cerdeira, A., Almeida, F., Matos, T. y Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553. DOI: https://doi.org/10.1016/j.dss.2009.05.016.

UCI Machine Learning Repository. *Wine Quality*. https://archive.ics.uci.edu/dataset/186/wine+quality. La ficha permite localizar procedencia, variables y escala; los conteos de este informe se comprobaron en el CSV local.

Scikit-learn. *Ridge*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html. Convención del objetivo de regularización.

Scikit-learn. *Support Vector Regression*. https://scikit-learn.org/stable/modules/svm.html#svr. Objetivo y formulación de SVR.

Scikit-learn. *Decision trees Regression criteria*. https://scikit-learn.org/stable/modules/tree.html#regression-criteria. Definición normalizada del criterio cuadrático.

Polars. *User guide*. https://docs.pola.rs/. Fundamento del motor tabular y su interfaz Python.

Rodríguez Bilbao, P. Material de la asignatura sin fecha editorial indicada: *Preprocesamiento en Machine Learning*, *Validación en Machine Learning*, *Aprendizaje supervisado Regresión*, *Aprendizaje supervisado Árboles de decisión*, *Aprendizaje supervisado Deep Learning* y *Práctica del primer parcial*. En esta última, archivo `Prac1PARDML22026.pdf`, p. 1, se verifican estructura del informe, límite del marco teórico y requisitos. La síntesis de lectura identifica los seis archivos individualmente.

Las referencias web se consultaron el 16 de septiembre de 2026; sus páginas `stable` no sustituyen el registro de versiones de la ejecución.
