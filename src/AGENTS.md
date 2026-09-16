# Desarrollo

Leer primero el AGENTS.md de la raíz y la teoría documentada. Mantener una clase por archivo, anotaciones completas y nombres que expliquen su intención. No añadir comentarios que repitan el código.

Cada caso tiene una responsabilidad. Los casos dependen de contratos de `shared/ports`; los adaptadores contienen Polars y scikit-learn. No introducir dependencias de interfaz gráfica en el cálculo. Preferir composición a herencia. No duplicar el motor de evaluación para crear el caso 03: conservar la estructura y reutilizar sus contratos.

No transformar `quality` en categorías, recortar valores ni cambiar modelos, particiones o métrica de selección sin actualizar la justificación y los resultados. Toda transformación aprendida se ajusta dentro del fold. La prueba se usa exclusivamente para el ganador elegido por validación y para la referencia.

Ejecutar pytest, ruff y mypy antes de declarar una implementación verificada. Registrar cambios de semillas, grillas y versiones.
