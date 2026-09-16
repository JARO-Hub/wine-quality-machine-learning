# Decisión 002 Herramientas y memoria

Polars lee y valida la tabla. Su núcleo está escrito en Rust y permite expresar operaciones por columnas. Aquí se lee un archivo pequeño completo, se conserva una copia cruda y se convierte una vez a matrices NumPy para scikit-learn. No se atribuye un ahorro medido a Rust: cambiar el lenguaje de implementación no demuestra por sí solo menor memoria total.

scikit-learn aporta estimadores y Pipeline que reproducen las formulaciones revisadas del libro. Se prefieren implementaciones verificadas a reescribir optimizadores en Rust para esta práctica. La biblioteca usa Python y componentes compilados; los seis modelos no pasan a ser modelos escritos en Rust por cargar con Polars.

Las matrices usan float64. Se registran sus bytes de datos con `nbytes`; ese dato excluye objetos Python, tablas Polars, copias de particiones, estructuras de árboles y caché de SVR. No representa RAM máxima del proceso. Ejecutar búsquedas y bosques con un solo trabajo evita multiplicar procesos para un dataset que no lo necesita.

Ruff y uv son herramientas del ecosistema Rust para revisar código y resolver dependencias, respectivamente. mypy comprueba los tipos de nuestro código; la frontera dinámica de scikit-learn se encapsula en un adaptador. Ninguna herramienta elimina la necesidad de comprobar el protocolo estadístico.

El visor usa una plantilla HTML local a partir de report.json. No entrena ni copia fórmulas de métricas. Funciona sin servicios externos y permite revisar rápidamente la clasificación por validación y los errores de prueba. La salida principal sigue estando disponible en consola.

Fuentes oficiales consultadas el 16 de septiembre de 2026: [Polars](https://docs.pola.rs/), [uv](https://docs.astral.sh/uv/), [Ruff](https://docs.astral.sh/ruff/), [scikit-learn](https://scikit-learn.org/stable/).
