# Lectura de las fuentes para el proyecto de calidad del vino

Esta síntesis explica qué aporta cada documento, cómo se relacionan sus ideas y qué decisiones tomamos para el proyecto. Se leyeron completos los seis PDF docentes. Del libro de Géron se revisaron secciones pertinentes de los capítulos 1, 2, 4, 5, 6 y 7; no se afirma haber leído sus 851 páginas completas. Las fórmulas y láminas relevantes se comprobaron también visualmente.

En los materiales docentes citamos la página del PDF, contando la portada como 1. Para el libro indicamos página impresa y página PDF: en los pasajes usados, la segunda tiene un desplazamiento de 30 páginas. Las propuestas aplicadas al vino se identifican como decisiones nuestras, distintas del contenido de las fuentes.

## Fuentes revisadas

| Fuente | Archivo o identificación | Páginas |
| --- | --- | ---: |
| Preprocesamiento | `1.4 Machine learning-datos.pdf` | 17 |
| Validación | `1.5MAchine learning-validacion.pdf` | 12 |
| Regresión | `1.6Aprendizaje supervisadoregresion.pdf` | 13 |
| Árboles y SVM | `1.7Aprendizaje supervisado-arboles.pdf` | 14 |
| Redes neuronales | `1.8 Aprendizaje supervisado-RNA2.pdf` | 18 |
| Práctica del primer parcial | `Prac1PARDML22026.pdf` | 2 |
| Libro de Géron | *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, segunda edición, 2019 | 851 |

Los seis primeros documentos corresponden al material de Patricia Rodríguez Bilbao. Todos se localizaron en la carpeta `MAchine Learning` suministrada. El libro funciona como desarrollo matemático y experimental; los apuntes organizan el recorrido de la asignatura.

## La práctica pide una progresión demostrable

La consigna propone avances: preparación de datos, implementación de modelos y validación. En su página PDF 1 limita el marco teórico a una página y sitúa problema, tecnologías, solución, código y pruebas dentro de Ingeniería. Esto favorece un informe breve de fundamentos y un desarrollo aplicado donde cada paso explique su función.

El primer avance pide tres tipos de procesos de limpieza, pero no los identifica inequívocamente. El apunte de datos distingue limpieza, normalización y reducción en PDF 5, y problemas de incompletitud, ruido e inconsistencia en PDF 8. Nuestra decisión es mostrar lo revisado y justificar lo aplicado. No necesitamos imputar cuando no hay faltantes ni eliminar variables para aparentar transformaciones.

La consigna final incluye cuatro parámetros de validación y matriz de confusión. Para calidad numérica proponemos MAE, MSE, RMSE y R², aclarando que MSE y RMSE están relacionados. La matriz requiere definir una tarea de clasificación; no se satisface convirtiendo puntuaciones en clases arbitrarias. Esa parte queda explícitamente pendiente. Los SVR se añaden por la petición actual del proyecto.

## Preparar datos significa preservar su significado

El material de preprocesamiento parte de registros y atributos, identifica problemas de calidad y después aborda transformación y selección de características (PDF 3–14). Esa secuencia separa dos preguntas: si una medición es confiable y cómo conviene representarla. Escalar una columna cambia su unidad de cálculo; no demuestra que se haya eliminado ruido ni convierte automáticamente su distribución en normal.

Géron hace visible este razonamiento con viviendas: el total de habitaciones resulta más interpretable al relacionarlo con hogares. Primero explica el significado del cociente y luego estudia su utilidad; no introduce combinaciones sólo porque puedan calcularse (pp. 61–64; PDF pp. 91–94). La idea transferible es justificar cada transformación y repetirla de forma consistente en nuevos datos.

Para vinos, mantenemos los once predictores, verificamos nulos y quitamos repeticiones exactas. Un extremo químico puede ser real, por lo que el IQR sirve como señal de revisión. La estandarización se aplica a modelos lineales y SVR porque sus penalizaciones o distancias dependen de la escala; sus medias y desviaciones se aprenden sólo con entrenamiento (Géron, pp. 69–70; PDF pp. 99–100). Esta es nuestra aplicación del criterio, no una obligación de transformar cualquier dataset igual.

## Validar permite distinguir aprendizaje y memorización

El apunte de validación define generalización como rendimiento en nuevas muestras y explica subajuste y sobreajuste (PDF 2–7). El sesgo expresa error sistemático; la varianza, sensibilidad a cambiar las muestras de entrenamiento. Aquí «sesgo» no significa intercepto, ni «varianza» simplemente dispersión de la columna calidad.

Géron presenta un árbol que llega a error cero en entrenamiento y después pierde su aparente ventaja al validar (pp. 73–74; PDF pp. 103–104). No trasladamos al vino el ranking de su ejemplo: trasladamos la necesidad de contrastarlo. También advierte que escoger parámetros mirando repetidamente prueba termina adaptando el sistema a ella (p. 31; PDF p. 61).

Por eso reservamos prueba, comparamos dentro de entrenamiento con cinco particiones y ajustamos allí las transformaciones. La media de RMSE organiza la selección; su desviación entre folds expresa dispersión, no un intervalo de confianza. La prueba final evalúa la decisión completa. Como el archivo ya fue explorado en el trabajo anterior, no presentamos esta evaluación como evidencia externa nueva.

## La regresión comienza por una hipótesis sencilla

El apunte de regresión representa una salida como pesos por entradas más intercepto y después introduce regularización (PDF 5–7). Géron desarrolla cómo ajustar esa hipótesis minimizando cuadrados y cómo usar pseudoinversa cuando una inversa ordinaria no existe (pp. 112–117; PDF pp. 142–147). El criterio de mínimos cuadrados no exige asumir normalidad para calcular la solución.

Nuestra secuencia comienza con OLS como referencia aditiva. Luego Ridge conserva la predicción lineal y penaliza coeficientes grandes: permite observar si restringir el ajuste mejora la validación. La normalización importa. El libro escribe `MSE+(α/2)‖w‖²`, mientras scikit-learn usa `SSE+α‖w‖²`; no se pueden copiar valores de `α` sin traducir esa convención (Géron, pp. 135–137; PDF pp. 165–167).

El apunte también incluye otros modelos, pero leerlos no obliga a incorporarlos. Elegimos dos candidatos que permitan explicar claramente qué cambia. Ridge reduce magnitudes de pesos; no garantiza eliminar variables ni obtener mejor rendimiento.

## Árboles y vectores de soporte necesitan un puente hacia regresión

Las láminas de árboles explican principalmente clasificación, categorías en hojas y criterios como Gini (PDF 3–7). En cambio, una hoja regresora predice la media porque esta minimiza la suma cuadrática dentro del grupo. Géron desarrolla ese paso en pp. 183–184, PDF pp. 213–214. Así evitamos aplicar fórmulas de categorías a una respuesta numérica.

El bosque agrega árboles obtenidos con aleatoriedad y promedia sus valores. Su motivación es que errores parcialmente distintos pueden compensarse; muchos árboles idénticos no aportarían esa ventaja (Géron, pp. 193–194 y 197–198; PDF pp. 223–224 y 227–228). Nuestra comparación entre CART y bosque mide el efecto de pasar de una partición a un conjunto de particiones diversas.

El mismo apunte introduce SVM clasificadora y kernels en PDF 8–13. Para SVR hace falta otra pieza: una banda que no penaliza errores hasta `ε` y cobra el exceso fuera de ella (Géron, pp. 162–164; PDF pp. 192–194). Comparamos SVR lineal con SVR RBF manteniendo esa pérdida. RBF expresa similitud según distancia y `γ` controla su alcance (pp. 159–161 y 171; PDF pp. 189–191 y 201). No cambiamos simultáneamente a etiquetas de clase.

## Redes neuronales aportan contexto sin ampliar el experimento

El PDF de redes presenta neuronas, activaciones y capas en páginas 3–8. Después introduce el problema de atribuir error a capas ocultas: conocemos la respuesta final deseada, pero no una etiqueta para cada neurona intermedia. La retropropagación distribuye la contribución al error usando derivadas (PDF 10–14). La diferencia entre entrenamiento e inferencia reaparece en PDF 16.

Esta lectura refuerza nuestro protocolo, aunque no incorpora una red entre los seis candidatos. Tampoco justifica necesitar GPU para una tabla de vinos. El título del libro incluye Keras y TensorFlow, pero sus capítulos iniciales ya sostienen el experimento acordado.
