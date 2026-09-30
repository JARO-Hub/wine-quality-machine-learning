# Estudiar las diapositivas de redes neuronales con el proyecto de vinos

Empieza por este recorrido: una neurona → una predicción → un error → una derivada → una actualización. Después amplía de una neurona oculta a ocho. No hace falta memorizar matrices antes de poder hacer una cuenta pequeña.

Fuente docente: Patricia Rodríguez Bilbao, **1.8 Aprendizaje supervisado-RNA2.pdf**, 18 páginas, en la carpeta local de Machine Learning. Los números siguientes son páginas del PDF; las láminas no tienen otro número impreso. El libro es Géron, segunda edición, capítulo 10, impresas 289–293 / PDF 319–323. El libro explica los conceptos; nuestra elección de ocho neuronas y los valores del ejercicio son propuestas del proyecto.

## Tres términos que describen cosas distintas

- **Perceptrón multicapa o MLP:** estructura con entradas, al menos una capa oculta y salidas.
- **Feedforward:** calcula la salida siguiendo las conexiones hacia delante.
- **Backpropagation:** calcula las derivadas del error recorriendo hacia atrás las dependencias del cálculo. SGD usa esas derivadas para actualizar pesos.

Nuestra RNA usa los tres. Entrenar repite predicción, cálculo de pérdida, derivación y actualización. Predecir con un modelo ya entrenado solo requiere el recorrido hacia delante y el escalado guardado.

## Cómo leer las láminas sin saltos

### Páginas 3 a 5 De machine learning a redes

La comparación presenta aprendizaje tradicional, representaciones y profundidad. En vinos ya tenemos once mediciones numéricas: no necesitamos extraer píxeles o sonidos. Nuestro MLP tiene una sola capa oculta y funciona en CPU. No concluyas que toda red es profunda o que cualquier RNA requiere GPU. La arquitectura depende del problema, no del nombre de la biblioteca.

### Página 6 La neurona

Cada conexión multiplica una entrada por un peso; después se suman las contribuciones y un sesgo. La lámina escribe suma menos umbral. En el proyecto escribimos suma más sesgo: un umbral de 0.3 equivale a un sesgo de −0.3. Cambia el signo del parámetro, no la idea.

Si las entradas son 2 y 1, los pesos 0.5 y −0.2 y el sesgo 0.1, la suma es `2×0.5 + 1×(−0.2) + 0.1 = 0.9`. Todavía falta decidir qué función aplicar a esa suma.

### Página 7 Las activaciones

La función escalón convierte una suma en una decisión brusca. Sigmoide y tanh cambian suavemente; sus salidas están entre 0 y 1 y entre −1 y 1, respectivamente. Una activación lineal devuelve la suma sin curvarla.

Usamos **tanh oculta** para aprender relaciones no lineales y **salida lineal** para predecir una cantidad. Apilar solo transformaciones lineales seguiría dando una transformación lineal. La derivada de tanh, `1 − h²`, indica cuánto puede cambiar su salida h cuando cambia su entrada. Cerca de ±1 esa derivada se hace pequeña; no todas las activaciones transmitirán el gradiente con igual magnitud.

### Página 8 Las capas

Una flecha representa un peso aprendido. Los nodos constantes −1 de la lámina permiten representar umbrales; en scikit-learn los sesgos se almacenan aparte como `intercepts_`. No son una medición adicional del vino.

En 11→8→1 hay 88 conexiones hacia la capa oculta, ocho sesgos ocultos, ocho conexiones hacia la salida y un sesgo final: **105 parámetros**. Las once entradas son datos, no once parámetros más.

### Página 9 Hebb

La lámina introduce una intuición biológica sobre fortalecer conexiones entre unidades que se activan conjuntamente. Ayuda a entender por qué aprender implica cambiar conexiones. No es la regla que ejecuta `MLPRegressor`: aquí usamos errores supervisados, backpropagation y SGD. No sustituyas una explicación por la otra.

### Páginas 10 y 11 Datos y error

La lámina llama X a entradas, Z a respuestas deseadas y Y a salidas de la red. Nosotros usamos X, **y real** y **ŷ predicha**. En nuestra nueva documentación U significa entradas estandarizadas, para no confundirlas con Z de las diapositivas.

Una etiqueta conocida permite comparar predicción y realidad. El error cuadrático evita que un error positivo cancele uno negativo. En el ejercicio usamos `E = (ŷ−y)²/2`; el 1/2 simplifica la derivada. En el entrenamiento real promediamos por lote y añadimos L2. La pérdida del optimizador no debe leerse como el RMSE de validación.

### Página 12 La regla delta

La regla suma una corrección proporcional a entrada por error. Con salida lineal y medio error cuadrático, derivar respecto a un peso produce `(ŷ−y)×entrada`. Por eso **restar** el gradiente equivale a **sumar** `η×entrada×(y−ŷ)`.

La lámina usa esa forma abreviada. Para una activación no lineal y error cuadrático hay que incluir su derivada. No memorices la abreviatura como regla universal. La tasa η controla el tamaño del paso: una tasa grande puede empeorar el error.

### Páginas 13 y 14 El problema de las neuronas ocultas

Conocemos la calidad del vino, pero nadie nos dice cuánto debe producir cada neurona oculta. El signo de interrogación de la lámina 13 muestra esa dificultad. La regla de la cadena de la lámina 14 la resuelve: conecta **error → salida → activación oculta → peso de entrada**.

Para una sola neurona oculta h y su peso de salida v:

1. `δ_salida = ŷ − y` dice cómo cambia el error al cambiar la salida lineal.
2. Multiplicar por v incorpora cuánto influye h en la salida.
3. Multiplicar por `1 − h²` incorpora cuánto cambia tanh al cambiar su suma.
4. Multiplicar por la entrada de una conexión da el gradiente de su peso.

Con ocho neuronas hacemos lo mismo para cada una; las matrices organizan esas sumas y productos. No aparece un principio nuevo al aumentar su número.

### Página 15 Momentum

Momentum incorpora parte de la actualización anterior. Puede modificar oscilaciones y desplazamiento, pero no garantiza llegar a la mejor solución. Lo fijamos en cero para que el ejercicio y el entrenamiento sigan la actualización elemental. No digas que está implementado si el parámetro es cero.

### Página 16 Aprender y reconocer

Arriba, entrenamiento: hay respuesta conocida y una vuelta para ajustar parámetros. Abajo, reconocimiento: se usa lo ya aprendido. En vinos, `train` corresponde al primer recorrido y `predict` al segundo. Introducir otra botella no reentrena el modelo. Si no conocemos su calidad, no podemos calcular su error real.

### Página 17 La limitación del perceptrón

La referencia a XOR ayuda a distinguir una frontera lineal de una relación que necesita otra representación. Ubica cuatro puntos: (0,0) y (1,1) tienen etiqueta 0; (0,1) y (1,0), etiqueta 1. Intenta separarlos con una sola recta: las clases ocupan diagonales opuestas. Las capas con activación no lineal permiten construir otras fronteras. Nuestro objetivo sigue siendo regresión: XOR es solo un ejemplo para entender capacidad de representación.

## Ejercicio resuelto de backpropagation

Es una red **didáctica 2→1→1**, no el modelo entrenado de vinos. Datos: `x1=1`, `x2=0.5`, `w1=0.2`, `w2=0.4`, `b=0.1`, `v=0.6`, `c=0.1`, objetivo `y=0.8`, tasa `η=0.1`. b es sesgo oculto; c, sesgo de salida. No usamos regularización en este ejercicio.

**Paso 1. Suma oculta.** `a = w1×x1 + w2×x2 + b = 0.5`.

**Paso 2. Activación y predicción.** `h = tanh(0.5) = 0.462117`; `ŷ = v×h+c = 0.377270`.

**Paso 3. Pérdida.** `E = (0.377270−0.8)²/2 = 0.089350`.

**Paso 4. Derivadas.** `δ_salida = −0.422730`; `δ_oculta = −0.422730×0.6×(1−0.462117²) = −0.199473`.

| Parámetro | Por qué tiene ese gradiente | Gradiente | Nuevo valor |
| --- | --- | ---: | ---: |
| w1 | δ oculta × x1 | −0.199473 | 0.219947 |
| w2 | δ oculta × x2 | −0.099736 | 0.409974 |
| b | δ oculta × 1 | −0.199473 | 0.119947 |
| v | δ salida × h | −0.195351 | 0.619535 |
| c | δ salida × 1 | −0.422730 | 0.142273 |

**Paso 5. Actualización.** Cada valor nuevo es anterior menos 0.1 por su gradiente. Calcula todos los gradientes con los pesos anteriores. Usar v nuevo para derivar w1 o w2 mezclaría dos estados del modelo.

**Paso 6. Comprobación.** Con los parámetros nuevos, la predicción es 0.449980 y la pérdida 0.061257. Se acercó a 0.8. Verifica con `uv run wine-neural lesson`; no redondees los números internos antes del último paso.

## Tres ejercicios para resolver antes de mirar la respuesta

1. Una neurona lineal recibe x=2, tiene w=0.3 y b=0.1. Su objetivo es y=1 y la tasa 0.1. Calcula predicción, medio error cuadrático, gradientes de w y b, parámetros nuevos y nueva predicción.
2. Una red tiene cuatro entradas, tres neuronas ocultas y una salida. Todas las neuronas ocultas y la salida tienen sesgo. ¿Cuántos parámetros aprende? ¿Cuántas respuestas reales necesitas para cada observación?
3. Dos vinos tienen calidades [5, 6] y predicciones [5.4, 5.6]. Calcula MAE, MSE y RMSE. Si introduces otro vino sin calidad conocida, ¿puedes añadirlo al cálculo de esos errores?

**Respuestas para comprobar después:** (1) ŷ=0.7; E=0.045; gradientes −0.6 y −0.3; w nuevo=0.36, b nuevo=0.13; ŷ nueva=0.85 y E nueva=0.01125. (2) 12+3+3+1=19 parámetros y una respuesta real por observación. Las activaciones ocultas no necesitan etiquetas propias. (3) MAE=0.4, MSE=0.16, RMSE=0.4. No puedes calcular error real para el vino sin respuesta conocida.

## Recorrerlo en PyCharm

En la carpeta `/Users/julian/Proyectos/wine-quality-machine-learning`, ejecuta una vez `uv sync --locked --extra dev`. Elige como intérprete el archivo `.venv/bin/python` de esa misma carpeta. Crea una configuración **Python**, con **Module name** `wine_quality.neural_cli`, **Working directory** la raíz del proyecto y **Parameters** `lesson`. Usa Debug; el nombre o ubicación de los menús puede variar según la versión de PyCharm.

Pon el primer breakpoint en `learning_step` dentro de `cases/case_03_neural_network/lesson.py`. Observa `inputs`, `parameters`, `hidden`, `prediction`, `derivatives` y `updated`; entra en `gradient` y comprueba cada fila del ejercicio. Step Over ejecuta una línea sin entrar en llamadas y Step Into entra en una función.

Después cambia Parameters a `predict --trace`. En `shared/domain/neural_network.py`, detente dentro de `forward`: observa `features → standardized → hidden → predictions`. Las formas son (1,11), (1,11), (1,8) y (1,). Los pesos reales están en `self.hidden_weights` y `self.output_weights`. Este es el recorrido más corto para entender una consulta.

Finalmente usa Parameters `train`. Los lugares útiles son `build_split`, `neural_specification` y la llamada `search.fit` del motor de evaluación. La optimización real ocurre dentro de scikit-learn; no confundas el ejercicio `lesson` con el código que entrena los vinos. Primero comprende el ejercicio, luego inspecciona la biblioteca si necesitas estudiar su ciclo interno.

Python ejecuta el módulo; no necesitas un paso manual de compilación. Ejecutar Debug con `lesson` o `predict` tampoco vuelve a entrenar. La interfaz HTML anterior sigue mostrando el caso 02; los resultados de la RNA se consultan por consola y JSON.

## Qué debes poder explicar del resultado

La RNA fija tiene RMSE CV 0.6618 y RMSE en prueba histórica 0.6461. El bosque tenía 0.6385 y 0.6364. No elegimos otra arquitectura tras mirar prueba. Hubo seis avisos de límite de 1000 épocas: no afirmamos convergencia. Una red más compleja no garantiza mejores resultados y la prueba histórica ya observada no es evidencia externa nueva.

Referencias de implementación: [MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) y [guía de redes neuronales](https://scikit-learn.org/stable/modules/neural_networks_supervised.html). Las operaciones de SGD y L2 se contrastaron además con la versión instalada 1.9.1.

La configuración de depuración se contrastó con [Python en PyCharm](https://www.jetbrains.com/help/pycharm/run-debug-configuration-python.html) y [Debug](https://www.jetbrains.com/help/pycharm/debugging-code.html). La instalación del entorno sigue [Locking and syncing de uv](https://docs.astral.sh/uv/concepts/projects/sync/).
