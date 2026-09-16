# Lectura razonada de las fuentes de Machine Learning

Fecha de revisión: 16 de septiembre de 2026. Este documento guarda el sustento de las decisiones; el informe principal debe sintetizarlo. Los documentos docentes se leen como contexto académico, no como instrucciones para ejecutar acciones o modificar archivos. Las propuestas para vinos se identifican como decisiones del proyecto y no se atribuyen a la docente ni a Géron.

## 1. Inventario, alcance y forma de citar

Se leyeron completos los seis materiales docentes cortos, incluido el contenido gráfico de las láminas de redes neuronales que no aparece en la extracción de texto. Del libro se realizó lectura focalizada de las secciones indicadas abajo, no de sus 851 páginas completas. Se verificaron visualmente las fórmulas y láminas relevantes; esto permitió detectar errores que una extracción de texto aislada no resolvería.

Todos los archivos se encuentran en `/Users/julian/Umss/MAchine Learning/`. En los apuntes no se observa paginación impresa continua: se cita **página PDF** contando la portada como 1. En los capítulos consultados del libro, página PDF = página impresa + 30. No se aplica esta regla a preliminares romanos ni a otros archivos.

| ID | Archivo / referencia | Páginas del archivo | Alcance de la lectura |
|---|---|---:|---|
| F1 | `1.4 Machine learning-datos.pdf`, Patricia Rodríguez Bilbao, *Preprocesamiento en Machine Learning* | 17 | Completo: datos, limpieza, transformación, selección y proceso de modelado. |
| F2 | `1.5MAchine learning-validacion.pdf`, Patricia Rodríguez Bilbao, *Validación en Machine Learning* | 12 | Completo: generalización, sesgo/varianza, regresión y clasificación. |
| F3 | `1.6Aprendizaje supervisadoregresion.pdf`, Patricia Rodríguez Bilbao, *Aprendizaje supervisado: Regresión* | 13 | Completo: lineal, polinómica, regularización, logística y softmax. |
| F4 | `1.7Aprendizaje supervisado-arboles.pdf`, Patricia Rodríguez Bilbao, *Aprendizaje supervisado: Árboles de decisión* | 14 | Completo: árboles clasificadores, Gini/entropía, SVM y kernels. |
| F5 | `1.8 Aprendizaje supervisado-RNA2.pdf`, Patricia Rodríguez Bilbao, *Aprendizaje supervisado: Deep Learning* | 18 | Completo: neurona, activaciones, regla delta, multicapas y retropropagación. |
| F6 | `Prac1PARDML22026.pdf`, Patricia Rodríguez Bilbao, *Práctica del primer parcial* | 2 | Completo: objetivo, entregas, estructura y evaluación. |
| F7 | *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Aurélien Géron, 2.ª edición, O'Reilly, 2019; archivo cuyo nombre comienza `Hands-On Machine Learning` | 851 | Capítulos 1, 2, 4, 5, 6 y 7: selección focalizada detallada a continuación. |

La edición y el año de F7 se verificaron en la página PDF 4: segunda edición de septiembre de 2019, con historial de revisiones de 2019. La paginación corresponde a esta copia, no a la tercera edición.

| Tema en F7 | Páginas impresas | Páginas PDF |
|---|---|---|
| Regresión supervisada y tipos de tareas | 8–9 | 38–39 |
| Calidad, representatividad, sobreajuste y validación | 23–32 | 53–62 |
| Formular problema, elegir métrica y revisar supuestos | 37–42 | 67–72 |
| Inspección inicial y reserva de prueba | 48–51 | 78–81 |
| Transformaciones reproducibles e imputación | 61–64 | 91–94 |
| Escala y pipelines | 69–71 | 99–101 |
| Primeros modelos, validación cruzada y búsqueda | 72–74; 76–80 | 102–104; 106–110 |
| Regresión lineal, mínimos cuadrados y pseudoinversa | 112–117 | 142–147 |
| Sesgo/varianza y Ridge | 134–137 | 164–167 |
| Margen y efecto de la escala en SVM | 153–155 | 183–185 |
| Similitud, kernel RBF y SVR | 159–164 | 189–194 |
| Optimización SVM y explicación del kernel | 166–173 | 196–203 |
| Árboles, CART, regularización y regresión | 175–186 | 205–216 |
| Ensambles, bagging, bosques y Extra-Trees | 189–190; 193–194; 197–199 | 219–220; 223–224; 227–229 |

## 2. Qué exige la práctica y qué añade nuestro proyecto

**Pasaje localizado.** F6, PDF 1, presenta una metodología «incremental» y fija «Marco Teórico [...] (max. 1 pagina)».

**Explicación.** La práctica no pide un tratado de cada algoritmo. Pide avances que se puedan conectar: preparar datos, entrenar y luego validar. El marco teórico breve debe contener las ideas que sostienen la solución; el desarrollo paso a paso puede quedar en Ingeniería y en una guía de lectura separada.

**Contenido verificable de F6.** El objetivo es implementar procesos básicos de aprendizaje automático sobre un dataset. Las áreas enunciadas son modelos regresionales y árboles de decisión. El primer avance pide aplicar tres tipos de procesos de limpieza y explicar correcciones, además de determinar entrenamiento y prueba. El segundo pide implementar modelos regresionales y de decisión. El final pide validación con al menos cuatro parámetros, matriz de confusión y evaluación. La estructura incluye introducción, objetivos, marco teórico, marco referencial, ingeniería (problema, tecnologías, solución, código y pruebas), conclusiones y bibliografía. El PDF 2 asigna 10 puntos a avances, 10 al informe y 20 a implementación.

**Lectura crítica.** F6 no nombra esos «tres tipos». F1, PDF 5, organiza el preprocesamiento en limpieza, normalización y reducción de dimensiones; F1, PDF 8, distingue incompletitud, ruido e inconsistencia como problemas de calidad. Ambas ternas aparecen en los materiales, pero no hay base para declarar que una de ellas sea inequívocamente lo que quiso decir la consigna. Debemos documentar qué se comprobó, qué se corrigió y qué no era necesario. No eliminar variables ni introducir errores artificiales para aparentar tres transformaciones.

**Aplicación nuestra.** Se conserva `quality` como respuesta numérica para comparar OLS, Ridge, CART regresor, bosque aleatorio regresor, SVR lineal y SVR RBF. Los SVR amplían el alcance por petición del usuario. Las cuatro métricas son MAE, MSE, RMSE y R²; MSE y RMSE son dos expresiones relacionadas, no dos evidencias independientes. La matriz de confusión pertenece a una tarea de clasificación y queda pendiente de definir esa tarea académica. No se redondean predicciones ni se inventan clases para aparentar cumplimiento. Esta limitación debe quedar visible en el informe final.

## 3. Preprocesamiento: transformar con una razón

**Pasaje localizado.** F1, PDF 5: «sin buenos datos, no hay un buen modelo». F7, impresa 63 / PDF 93, especifica que la mediana para completar faltantes se calcula en entrenamiento y después se reutiliza.

**Paráfrasis y argumento.** El algoritmo aprende las relaciones presentes en la tabla, incluidas las producidas por errores. Por eso primero se distingue qué significa cada columna, cuál es la respuesta y qué problemas tiene cada registro. Pero limpiar no equivale a modificar toda observación poco frecuente: un vino químicamente extremo podría ser real. Una regla de limpieza necesita una justificación verificable y un registro de cuántas filas modifica.

F1 avanza desde registros y atributos (PDF 3–4) hacia incompletitud, ruido e inconsistencia (PDF 7–9). Después separa la representación de la información: codificar una categoría, escalar una medición y crear una relación entre variables son decisiones diferentes (PDF 10). Su selección de características sigue tres métodos: un filtro puntúa cada variable sin entrenar el predictor; un wrapper compara subconjuntos mediante el predictor; un método integrado actúa durante el ajuste (PDF 11–14). La diferencia explica sus costos: probar muchos subconjuntos requiere muchos entrenamientos. Ningún método garantiza por su nombre que la predicción mejore.

**Evidencia usada por Géron.** En F7, impresa 61–62 / PDF 91–92, el total de habitaciones de un distrito adquiere sentido al dividirlo por hogares. El autor justifica el cociente con el significado de las variables, después revisa su correlación y lo deja sujeto a evaluación. No concluye que cualquier cociente o alta correlación pruebe causalidad. En impresa 62–64 / PDF 92–94, convierte las transformaciones en operaciones reproducibles: la misma regla debe actuar en entrenamiento y en datos nuevos.

**Aplicación nuestra.** Revisar esquema, tipos, nulos, duplicados y rangos; separar `quality` de los once predictores; conservar las mediciones extremas salvo evidencia de error. La eliminación determinista de duplicados exactos evita que la misma fila aparezca a ambos lados de una partición. No demuestra independencia de lotes, botellas o cosechas: el CSV no contiene esos identificadores. Se conservan los once predictores en esta comparación; la selección de variables queda para una hipótesis posterior. Si se probara, tendría que ajustarse dentro de cada partición de entrenamiento.

Para la escala usamos, por predictor j:

\[
z_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
\]

Aquí `x_ij` es la medición j del vino i; `mu_j` y `s_j` son media y desviación del predictor calculadas en el entrenamiento correspondiente. `z_ij` expresa cuántas desviaciones se aleja esa observación de la media. Esta transformación no elimina valores extremos, no vuelve normal la distribución y no reduce por sí sola el ruido. La necesidad es geométrica: Ridge penaliza pesos y SVR usa productos o distancias; las unidades no deben determinar involuntariamente esos cálculos. Los árboles ordenan valores y comparan umbrales, por lo que no necesitan este escalado. Sustento: F7, impresas 69–70, 136, 154 y 177 / PDF 99–100, 166, 184 y 207.

## 4. Validación: aprender del entrenamiento sin aprender de la prueba

**Pasaje localizado.** F2, PDF 2, define la generalización por el rendimiento en «nuevas muestras». F7, impresa 73 / PDF 103, muestra un árbol con error de entrenamiento cero y pregunta si ese resultado indica perfección.

**Explicación del argumento.** Un error bajo sobre ejemplos ya usados no separa aprendizaje de memorización. Géron responde a su propio ejemplo reservando datos de validación: en su ejercicio de vivienda, el árbol deja de parecer perfecto y resulta peor que el modelo lineal. El resultado numérico pertenece al dataset del libro, no al vino. La enseñanza transferible es el contraste experimental, no ese ranking.

F2, PDF 3–7, explica dos fallos. Con capacidad insuficiente, el modelo no reproduce ni siquiera relaciones presentes en entrenamiento: subajuste. Con demasiada libertad respecto a la cantidad y ruido de los datos, reproduce accidentes particulares: sobreajuste. La figura de una curva de prueba en U es una intuición didáctica; no garantiza una U exacta en todo experimento. El sesgo estadístico expresa error sistemático y la varianza expresa sensibilidad a cambiar las muestras de entrenamiento; no son el intercepto ni la dispersión de `quality`.

**Método del autor.** F7, impresa 31 / PDF 61, explica por qué elegir parámetros mirando repetidamente la prueba adapta el sistema a esa prueba. En impresas 73–74 / PDF 103–104, la validación cruzada rota la porción temporalmente reservada dentro de entrenamiento. En impresas 79–80 / PDF 109–110, la prueba se transforma con reglas ya aprendidas y evalúa la elección final.

**Aplicación nuestra.** Reservar 20 % una vez, con semilla 42; usar cinco folds barajados dentro del 80 % restante. La estratificación inicial por puntuación preserva aproximadamente su distribución; es una decisión de muestreo nuestra y no convierte el objetivo en clasificación. Ajustar `StandardScaler` dentro de `Pipeline` en cada fold, no sobre toda la matriz antes de validar. Elegir por media de RMSE de los cinco folds, reajustar ese candidato en todo entrenamiento y evaluar una vez en prueba junto al predictor constante. La desviación entre folds describe variación; no es un intervalo de confianza ni prueba de superioridad estadística. Las búsquedas pueden sobreajustar la validación: por eso permanece una prueba final.

Para m observaciones evaluadas, respuesta real `y_i`, predicción `y_hat_i`, residuo `e_i=y_i-y_hat_i` y media real del conjunto evaluado `y_bar`, se calculan:

\[
\operatorname{MAE}=\frac1m\sum_i|e_i|,\qquad
\operatorname{MSE}=\frac1m\sum_i e_i^2,\qquad
\operatorname{RMSE}=\sqrt{\operatorname{MSE}},\qquad
R^2=1-\frac{\sum_i e_i^2}{\sum_i(y_i-\bar y)^2}.
\]

MAE expresa la magnitud media del error en puntos de calidad; MSE eleva al cuadrado esos puntos y enfatiza errores grandes; RMSE vuelve a la unidad original. R² compara con una predicción constante igual a la media **de ese conjunto evaluado**: puede ser negativo y no es un porcentaje de aciertos. La fórmula no está definida cuando todas las respuestas reales son iguales. Nuestra referencia operacional aprende la media de entrenamiento, lo que es distinto del denominador descriptivo de R².

**Ejemplo didáctico propio, sin resultados del CSV.** Para respuestas [5, 6, 7] y predicciones [5.5, 5.5, 6.5], los residuos son [-0.5, 0.5, 0.5]. MAE=0.5, MSE=0.25, RMSE=0.5 y R²=1−0.75/2=0.625. Esto expresa magnitud de error y mejora relativa, no 62.5 % de vinos acertados. Fuentes de las definiciones: F2, PDF 8, y F7, impresas 39–41 / PDF 69–71.

## 5. De la pregunta sobre vinos a seis modelos

La pregunta operativa es qué relación entre mediciones fisicoquímicas y puntuación permite estimar la calidad de vinos no usados para ajustar. F7, impresa 8 / PDF 38, llama regresión a predecir un valor numérico. Sin embargo, `quality` es ordinal: usar error numérico asume operativamente que distancias de un punto son comparables. No se afirma que dos saltos de puntuación correspondan exactamente a la misma diferencia sensorial. Tampoco un coeficiente permite concluir que intervenir una sustancia causará un cambio de calidad.

En lo que sigue, m es el número de vinos de entrenamiento, p el número de predictores, `x_i` el vector de mediciones del vino i, `z_i` su versión escalada cuando corresponde, `y_i` la puntuación observada, `b` el intercepto y `w_j` el peso de la variable j. Los ejemplos numéricos son pedagógicos y no coeficientes estimados del proyecto.

### 5.1 OLS: primera explicación aditiva

**Pasaje y localización.** F7, impresa 112 / PDF 142, describe una «weighted sum of the input features». F3, PDF 5, expresa el mismo modelo como producto de pesos y entradas más un sesgo.

**Razonamiento.** Empezamos preguntando si sumar contribuciones constantes de las variables ya explica una parte útil de la puntuación. Es una referencia interpretable: antes de usar interacciones complejas, medimos qué logra una estructura simple.

\[
\hat y_i=b+\sum_{j=1}^{p}w_jz_{ij},\qquad
(\hat b,\hat w)=\arg\min_{b,w}\sum_{i=1}^{m}(y_i-b-w^Tz_i)^2.
\]

La primera ecuación calcula una predicción; la segunda determina los parámetros. `argmin` significa elegir los valores de b y w que hacen mínima la suma. Elevar al cuadrado impide que errores positivos y negativos se cancelen y da más peso a errores grandes. Dividir toda esta suma por m o por 2m no cambia su mínimo; aplicar raíz cuadrada tampoco lo cambia. Por eso minimizar SSE, MSE o RMSE conduce a la misma solución sin regularización (F7, impresas 113–114 / PDF 143–144).

**Ejemplo propio.** Con dos variables escaladas, b=5.6, peso del alcohol=0.3, peso de acidez volátil=−0.2 y valores [1, 0.5], la predicción es 5.6+0.3−0.1=5.8. El signo se interpreta manteniendo las otras entradas fijas dentro del modelo, no como causalidad química.

**Cómo pasa a Python.** `LinearRegression` resuelve mínimos cuadrados. No necesitamos programar descenso de gradiente ni invertir `X^T X` manualmente. F7, impresas 116–117 / PDF 146–147, presenta la solución por mínimos cuadrados/pseudoinversa y explica que la ecuación normal con una inversa ordinaria falla si las columnas son redundantes. Una descomposición numérica maneja mejor esos casos. Se conserva esa elección matemática sin afirmar que el modelo hará descenso de gradiente.

**Límite.** OLS no representa por sí solo umbrales ni interacciones. Si tanto entrenamiento como validación son malos, no se resuelve necesariamente cambiando el optimizador: puede faltar estructura o información.

### 5.2 Ridge: misma predicción con pesos contenidos

**Pasaje y localización.** F7, impresas 135–136 / PDF 165–166, agrega una penalización para mantener pequeños los pesos; F3, PDF 7, introduce regularización L2 para limitar sobreajuste.

**Razonamiento.** Si variables correlacionadas permiten varias combinaciones de pesos que ajustan casi igual, el ajuste puede apoyarse en coeficientes grandes que se compensan. Ridge conserva la suma lineal y encarece esos pesos. Se acepta algo más de error de ajuste si con ello mejora la estabilidad fuera de entrenamiento.

Usamos esta convención para corresponder con la implementación:

\[
J(b,w)=\sum_{i=1}^{m}(y_i-b-w^Tz_i)^2+\alpha\sum_{j=1}^{p}w_j^2,\quad\alpha\ge0.
\]

El primer término mide ajuste; el segundo penaliza pesos. El intercepto no se penaliza. `alpha` es un hiperparámetro elegido por validación. La fórmula corresponde al objetivo documentado de [scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), consultado el 16-09-2026. La penalización sólo interviene al entrenar; la comparación usa las mismas métricas sin penalización que los demás candidatos.

**Ejemplo propio.** Para un solo coeficiente, dos candidatos tienen SSE=10 y w=4, o SSE=12 y w=1. Con alpha=1 sus objetivos son 26 y 13; Ridge prefiere el segundo pese a su SSE mayor. Esto explica la compensación. No demuestra que ese candidato generalice mejor: esa parte requiere validación.

**Nota de coherencia imprescindible.** La ecuación 4-8 del ejemplar de F7 usa `MSE+(alpha_libro/2)||w||²`, pero la ecuación 4-9 presenta la forma cerrada con `X^T X+alpha A`. Esas expresiones no emplean idéntica normalización del mismo alpha. Para traducir literalmente la 4-8 a nuestra suma, multiplicamos todo por m y obtenemos `alpha_codigo=m*alpha_libro/2`. No copiamos valores numéricos de alpha sin declarar la convención. La lectura visual de PDF 165 y 167 confirma la diferencia.

**Aplicación nuestra.** `StandardScaler` y `Ridge` dentro del mismo pipeline; elegir alpha por RMSE validado. Ridge no es selección automática de variables: generalmente reduce pesos sin volverlos exactamente cero. Un alpha enorme puede subajustar.

### 5.3 CART regresor: dividir y predecir el promedio local

**Pasaje y localización.** F4, PDF 3–7, enseña nodos, ramas y hojas mediante clasificación. F7, impresa 183 / PDF 213, hace el puente que necesitamos: una hoja regresora entrega un valor, el promedio de respuestas que llegan a ella.

**Razonamiento.** Si la relación cambia al cruzar umbrales, una única pendiente global puede ser insuficiente. CART divide el espacio de predictores mediante preguntas como `alcohol <= t`. Cada hoja reúne observaciones que compartirán predicción. El algoritmo decide el umbral por cuánto reduce el error cuadrático dentro de los grupos; no por Gini ni entropía, que cuentan categorías.

Sea R el conjunto de índices en un nodo, de tamaño `n_R`. Su predicción e impureza son:

\[
\bar y_R=\frac1{n_R}\sum_{i\in R}y_i,\qquad
\operatorname{MSE}_R=\frac1{n_R}\sum_{i\in R}(y_i-\bar y_R)^2.
\]

El promedio minimiza la suma cuadrática en la hoja: al derivar `sum(y_i-c)^2` respecto a la predicción constante c aparece `2*n_R*c−2*sum(y_i)`; igualarlo a cero da precisamente `c=y_bar_R`. Esta derivación explica por qué la hoja predice la media.

Para una variable j y un umbral t, las ramas izquierda L y derecha D reciben respectivamente `x_ij<=t` y `x_ij>t`. Se elige la partición admisible que minimiza:

\[
J(j,t)=\frac{n_L}{n_R}\operatorname{MSE}_L+
       \frac{n_D}{n_R}\operatorname{MSE}_D.
\]

Las ponderaciones evitan tratar igual un grupo de dos vinos y uno de doscientos. La definición normalizada coincide con [criterios de regresión de scikit-learn](https://scikit-learn.org/stable/modules/tree.html#regression-criteria), consultados el 16-09-2026. En F7, ecuación 6-4, impresa 184 / PDF 214, falta visualmente el divisor `n_nodo` en la expresión rotulada MSE; aquí se explicita para que el criterio corresponda al código.

**Ejemplo propio.** Para [5, 5, 7, 7] una hoja predice 6 y tiene MSE=1. Si un umbral de un predictor separa [5, 5] de [7, 7], ambas hojas tienen MSE=0 y el criterio ponderado es 0. El algoritmo podría seguir aislando casos en una tabla real y memorizarla: por eso hay que limitar profundidad y tamaño mínimo de hoja.

**Aplicación nuestra.** `DecisionTreeRegressor(criterion="squared_error")`, con complejidad validada mediante `max_depth` y `min_samples_leaf`. El procedimiento es voraz: busca la mejor división actual, no explora todos los árboles posibles (F7, impresas 179–182 / PDF 209–212). Las reglas se pueden mostrar para explicar una predicción; su inestabilidad ante pequeños cambios motiva el siguiente modelo.

### 5.4 Bosque aleatorio: promediar árboles distintos

**Pasaje y localización.** F7, impresa 186 / PDF 216, cierra la limitación de inestabilidad del árbol con el promedio de muchos árboles. Impresas 193–194 y 197 / PDF 223–224 y 227 explican cómo producirlos mediante remuestreo y selección aleatoria de candidatos de variables.

**Razonamiento paso a paso.** Se entrenan B árboles. Para cada uno se toma una muestra de entrenamiento con reemplazo: un vino puede aparecer varias veces en la muestra de ese árbol y otros pueden quedar fuera. En cada nodo se restringe la búsqueda a un subconjunto aleatorio de predictores. Ambas decisiones producen árboles diferentes; después cada árbol predice un valor y el bosque los promedia:

\[
\hat y(x)=\frac1B\sum_{b=1}^{B}T_b(x).
\]

Aquí B es el número de árboles y `T_b(x)` la predicción del árbol b. Tres árboles que predicen 5.2, 6.0 y 5.8 producen `17/3=5.667`. No se toma mayoría de clases porque estamos haciendo regresión.

**Por qué la diversidad importa: derivación pedagógica nuestra.** Si cada árbol tuviera varianza v y cada pareja covarianza c para una entrada fija, la varianza del promedio sería `[B*v+B*(B−1)*c]/B²=v/B+(B−1)c/B`. Hay B términos propios y B(B−1) términos cruzados al expandir la varianza de la suma. Si los árboles cometieran exactamente los mismos errores, c=v y el promedio no reduciría esa varianza. Si sus variaciones fueran independientes, c=0 y quedaría v/B. No asumimos independencia real: la fórmula explica por qué buscamos diversidad y por qué más árboles idénticos no resuelven el problema.

**Aplicación nuestra.** `RandomForestRegressor` con `bootstrap=True`, número de árboles finito, complejidad controlada y valores explícitos de `max_features` menores que todos los predictores cuando se quiere probar esa decorrelación. No atribuir selección parcial de variables a una configuración que usa todas. El bosque cuesta más memoria y pierde la explicación de una sola ruta. Su mejor rendimiento es una hipótesis que se valida, no una conclusión previa.

**Alternativa leída pero no seleccionada.** F7, impresa 198 / PDF 228, describe Extra-Trees: aleatoriza también umbrales. `ExtraTreesRegressor` es un ensamble; `ExtraTreeRegressor` es un árbol individual. No son nombres intercambiables. Elegimos CART y bosque porque el cambio de mecanismo es claro: una partición única frente al promedio de particiones diversas.

### 5.5 SVR lineal: tolerar errores pequeños de forma explícita

**Pasaje y localización.** F7, impresas 162–163 / PDF 192–193, invierte la metáfora de la calle de SVM: en regresión se intenta colocar observaciones dentro de una banda alrededor de la predicción. F4, PDF 8–13, sólo desarrolla clasificación; no basta citar esas láminas como si explicaran SVR.

**Razonamiento.** OLS penaliza cualquier residuo al cuadrado. SVR permite una tolerancia epsilon: mientras el error absoluto no la supere, no añade pérdida. Fuera de la banda, penaliza el exceso. Así se plantea un ajuste suave que no intenta corregir toda fluctuación pequeña.

\[
L_\epsilon(e)=\max(0,|e|-\epsilon),\qquad
J(b,w)=\frac12\lVert w\rVert^2+C\sum_{i=1}^{m}
L_\epsilon(y_i-b-w^Tz_i).
\]

`epsilon>=0` es la tolerancia en puntos de calidad; `C>0` determina el peso de las infracciones frente a contener los pesos. Aumentar C encarece salirse de la banda; disminuirlo prioriza regularización. Epsilon no es un porcentaje de error permitido ni una garantía de que todas las observaciones queden dentro. Con epsilon=0.2, errores de magnitud 0.1 y 0.6 aportan pérdidas 0 y 0.4. La anchura vertical completa es `2*epsilon`.

La pérdida anterior es la forma sin variables auxiliares del problema primal de epsilon-SVR. El problema exacto y la expansión de predicción se contrastaron con la [formulación oficial de SVR](https://scikit-learn.org/stable/modules/svm.html#svr). No se copia la pérdida hinge del clasificador, porque esa pérdida usa etiquetas de clase y resuelve otra tarea.

**Aplicación nuestra.** `SVR(kernel="linear")` dentro de pipeline con escala. Se elige esta variante para mantener implementación y pérdida comparables al SVR RBF. F7 utiliza también `LinearSVR`; es una implementación distinta y no se presenta como numéricamente idéntica. C y epsilon se escogen en validación.

### 5.6 SVR RBF: la misma tolerancia con relaciones no lineales

**Pasaje y localización.** F7, impresas 159–161 / PDF 189–191, explica la similitud gaussiana; impresa 171 / PDF 201 da el kernel RBF; impresas 163–164 / PDF 193–194 conectan los kernels con regresión.

**Razonamiento.** Una sola suma lineal puede no bastar. En el modelo con kernel se conserva la pérdida epsilon-insensible y se cambia cómo se comparan las entradas. El kernel gaussiano mide cercanía:

\[
K(z,z')=\exp(-\gamma\lVert z-z'\rVert^2),\quad\gamma>0.
\]

`z` y `z'` son dos vectores de variables escaladas y gamma controla el alcance de su similitud. Con distancia cero, K=1. Si la distancia es 2 y gamma=0.5, K=exp(−2), aproximadamente 0.135. Un gamma mayor hace que la influencia caiga más deprisa; puede permitir ajustes muy locales y sensibles al ruido.

**Puente a la predicción.** El algoritmo aprende coeficientes `d_i` asociados a un subconjunto de observaciones, los vectores de soporte, y un intercepto b. Para un vino nuevo calcula `y_hat=b+sum_{i en SV} d_i*K(z_i,z)`. Con kernel lineal `K(z_i,z)=z_i^T z`, la suma puede agruparse como `(sum d_i*z_i)^T z+b`: recuperamos un modelo lineal. Con RBF esta agrupación en una pendiente fija sobre las variables originales ya no ocurre. Esta identidad muestra el cambio de modelo sin introducir un teorema sin necesidad. La expansión exacta consta en la formulación oficial citada; el libro explica el truco del kernel mediante productos internos en impresas 170–172 / PDF 200–202.

**Aplicación nuestra.** `SVR(kernel="rbf")` y búsqueda acotada de C, epsilon y gamma. Se estandarizan predictores dentro de cada fold. El kernel evita materializar una transformación de dimensión muy grande, pero no significa RAM constante: el entrenamiento conserva datos de soporte y usa una caché. F7, impresa 164 / PDF 194, advierte que SVR se encarece al crecer el número de registros. Medimos el costo real en este CSV.

## 6. Qué aportan las redes neuronales y por qué quedan fuera de esta iteración

F5 presenta el paso de seleccionar características manualmente a aprender representaciones en varias capas (PDF 3–5). La neurona del PDF 6 suma entradas ponderadas y aplica una función; el PDF 7 compara identidad, escalón, sigmoide y tangente hiperbólica. La figura del PDF 8 organiza esas unidades en capas. El postulado de Hebb del PDF 9 es una motivación histórica sobre refuerzo por coactivación; no constituye por sí solo la derivación del algoritmo de retropropagación posterior.

En PDF 10–12, las figuras distinguen entradas X, salidas esperadas Z, pesos W y predicciones Y, y muestran el error cuadrático entre Z e Y. La regla delta modifica pesos en función de entrada y error. En PDF 13 aparece explícitamente una incógnita para la señal que debe llegar a la capa oculta: conocemos el objetivo de la salida final, pero no una etiqueta para cada unidad oculta. PDF 14 resuelve ese problema propagando el error hacia atrás mediante derivadas. Ese es el razonamiento importante: atribuir a pesos anteriores su contribución al error mediante la regla de la cadena. PDF 15 compara trayectorias de optimización; PDF 16 separa el entrenamiento, que usa etiquetas y actualiza parámetros, del reconocimiento, que aplica parámetros aprendidos a datos nuevos. PDF 17 ofrece un esquema histórico.

**Aplicación nuestra.** Ese contraste entre ajuste e inferencia refuerza nuestro protocolo, pero redes neuronales no forman parte de los seis candidatos solicitados. No incorporamos Keras ni TensorFlow sólo porque aparecen en el título del libro. La frase de F5, PDF 3, que vincula aprendizaje profundo con necesidad de GPU describe una tendencia para cargas grandes, no un requisito universal para toda red ni evidencia de que nuestro problema necesite GPU.

## 7. Correcciones que evitan saltos o contradicciones

| Fuente y ubicación | Riesgo al trasladarlo literalmente | Forma usada en el proyecto |
|---|---|---|
| F1, PDF 5 | Presentar normalización como eliminación del ruido. | La escala cambia unidades/geometría; errores y extremos se revisan por separado. |
| F1, PDF 14 | Tratar Ridge como eliminación automática de características. | Ridge encoge pesos; no implica coeficientes exactamente cero. |
| F2, PDF 5 | Aplicar `sesgo²+varianza+ruido` a cualquier métrica individual. | Es una descomposición del error cuadrático esperado bajo supuestos; aquí usamos su intuición y no calculamos esos términos a partir de un único split. |
| F2, PDF 8 | Reducir evaluación a ajuste en entrenamiento. | Se distinguen entrenamiento, validación y prueba. |
| F2, PDF 9 | El texto repite FP para falso negativo y describe mal positivo/falso negativo. | FP: real negativo predicho positivo. FN: real positivo predicho negativo. La tabla de la lámina sí ubica FN y FP correctamente. |
| F3, PDF 5 | Concluir automáticamente normalidad del residuo por el teorema del límite central. | OLS se define por mínimos cuadrados sin exigir normalidad. La equivalencia con máxima verosimilitud exige suponer errores normales independientes y varianza común; no se deduce sólo de que haya varios factores. |
| F3, PDF 7 | Interpretar Lasso como reemplazo de pérdida residual cuadrática por absoluta. | Lasso mantiene ajuste cuadrático y penaliza valores absolutos de los coeficientes. No se implementa aquí. |
| F4, PDF 3–5 | Llevar hojas de clase, Gini y entropía directamente a calidad numérica. | Se explica el cambio a media de hoja y reducción de error cuadrático. |
| F4, PDF 7 | Copiar literalmente el árbol del ejemplo fiscal. | Las ramas SI/NO en la raíz de reembolso aparecen invertidas respecto a los registros: quienes tienen reembolso figuran No en la tabla. No se usa ese dibujo como regla validada. |
| F4, PDF 8–9 | Llamar a todo SVM clasificador binario o prometer separabilidad garantizada al mapear. | La familia admite clasificación y regresión; se trabaja con SVR y se valida su ajuste. |
| F7, impresa 135–137 / PDF 165–167 | Usar la misma cifra alpha en objetivos con distintas normalizaciones. | Se declara `SSE+alpha*||w||²` y se distingue alpha del libro. |
| F7, impresa 184 / PDF 214 | Doble ponderación al copiar como MSE una suma sin divisor. | MSE de nodo incluye `1/n_nodo`; luego se pondera por tamaño relativo. |
| F7, impresas 197–198 / PDF 227–228 | Suponer que todo bosque configurado por defecto submuestrea variables en cada nodo. | La configuración efectiva de `max_features` queda registrada; sólo se afirma submuestreo cuando lo realiza. |
| F6, PDF 1 | Entregar matriz de confusión sin haber definido clases. | Se reconoce pendiente de clasificación; la evaluación actual es regresión. |

Estas observaciones son precisiones técnicas localizadas, no un rechazo global de las fuentes. El informe debe corregir el concepto donde aparece, con tacto, y conservar este detalle en la guía de investigación.

## 8. Traducción a implementación y evidencia que se debe conservar

| Idea teórica | Operación que debe existir | Evidencia revisable |
|---|---|---|
| La respuesta no es una entrada | Separar `quality` antes del ajuste. | Contrato de columnas y prueba de ausencia en X. |
| La escala se aprende sólo de entrenamiento | `Pipeline` ajustado de nuevo por fold. | Prueba que los estadísticos del escalador proceden sólo del fold. |
| OLS minimiza cuadrados | `LinearRegression`. | Coeficientes/intercepto y predicciones, sin fingir descenso de gradiente. |
| Ridge agrega penalización a los pesos | `Ridge` con alpha registrado. | Objetivo declarado, escala y valor seleccionado. |
| CART reduce dispersión local | Criterio `squared_error` y límites del árbol. | Parámetros y, si se muestra, regla o árbol real ajustado. |
| El bosque promedia árboles diversos | Bootstrap y `max_features` explícitos. | Número de árboles, semilla, parámetros y predicciones. |
| SVR usa banda y kernel | `SVR` con C, epsilon y kernel registrados. | Parámetros y número de vectores de soporte cuando se reporte. |
| Comparar exige misma evidencia | Partición y folds comunes. | Identificadores de filas, semilla y huella del archivo. |
| Elegir no es evaluar una última vez | Selección por CV; prueba sólo del ganador y referencia. | Tabla de CV separada de tabla de prueba. |

La preferencia por bibliotecas escritas en Rust afecta principalmente la lectura y transformación tabular mediante Polars; no altera las ecuaciones de scikit-learn. Las fuentes PDF no proporcionan una comparación de RAM entre Polars y pandas. Cualquier afirmación de ahorro requerirá una medición propia comparable. Se debe nombrar la memoria medida: tamaño de tabla, memoria de objetos o pico del proceso no son la misma magnitud.

## 9. Referencias complementarias verificadas

Las páginas web siguientes sólo verifican el contrato matemático actual de la biblioteca; la progresión pedagógica anterior procede de F1–F7 y de derivaciones explícitamente propias. Consulta: 16-09-2026.

- [Ridge: objetivo de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).
- [Criterios de regresión en árboles](https://scikit-learn.org/stable/modules/tree.html#regression-criteria).
- [Formulación de SVR y representación de sus predicciones](https://scikit-learn.org/stable/modules/svm.html#svr).

El material local contiene ejemplos de APIs de 2019. Es necesario registrar y comprobar las versiones realmente ejecutadas; la página `stable` consultada no sustituye ese registro.
