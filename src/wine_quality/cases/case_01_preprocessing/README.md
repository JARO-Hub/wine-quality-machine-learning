# Caso 01 Preparación

Entrada: CSV original con once predictores y quality. Salida: Dataset tipado, validado, deduplicado y con identificadores originales. La consola exporta clean.csv y report.json.

No se escala ni recorta antes de separar datos. Faltantes o conflictos entre predictores idénticos y distintas respuestas se reportan como error, porque tratarlos necesita una decisión explícita. El motor permite delimitadores coma y punto y coma.

El servicio depende de DatasetRepository; el adaptador Polars ejecuta lectura y validación. Los tests comprueban esquema, dominio, orden, eliminación exacta de repetidos y conservación de extremos. Leer AGENTS.md de la raíz y de src/ antes de cambiar esta política.
