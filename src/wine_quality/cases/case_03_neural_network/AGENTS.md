# Acuerdos de la iteración 03

Conservar el candidato fijado en docs/decisions/003-red-neuronal.md y la misma partición histórica. Cualquier estudio de otra arquitectura es una iteración nueva con protocolo previo, no una corrección silenciosa tras mirar prueba.

Distinguir tres operaciones: train aprende y evalúa; predict solo aplica escalas y pesos guardados; lesson ejecuta un ejercicio didáctico independiente. Nunca introducir la calidad conocida como predictor ni usarla para ajustar durante una consulta.

Mantener visibles los avisos de convergencia y la procedencia histórica de manual_example.json. El ejemplo fue introducido por el asistente para verificar el flujo, no es evidencia aportada por una persona ni una validación externa.

Al cambiar la fórmula de inferencia, comprobar equivalencia con Pipeline.predict. Al cambiar el ejercicio, comprobar sus derivadas numéricamente y actualizar simultáneamente guía e informe. No añadir interfaces o dependencias gráficas para reproducir tres comandos sencillos.
