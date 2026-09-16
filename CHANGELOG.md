# Registro de iteraciones

## 2026 09 16 Iteraciones 01 y 02

- Se revisaron el informe original, su código, la base de vinos y siete PDF, distinguiendo lectura completa de apuntes y lectura focalizada del libro.
- Se detectó que el recorte IQR modificaba una tabla diferente de la exportada. El caso 01 conserva los valores originales, elimina duplicados exactos y valida el esquema.
- Se fijó la comparación de OLS, Ridge, CART, bosque aleatorio, SVR lineal y SVR RBF con referencia constante. Se separó selección por validación de la prueba final.
- Se inició una arquitectura por casos, tipada y con pruebas de integridad experimental, acuerdos para agentes y documentación de fuentes.
- Pasaron 21 pruebas científicas, Ruff y mypy estricto. La auditoría independiente reconstruyó métricas desde las predicciones y verificó todas las particiones.
- La selección eligió el bosque aleatorio: RMSE CV 0.6385 y de prueba 0.6364, frente a 0.8191 de la referencia en prueba.
- Se incorporaron dos Word, una vista HTML local y un registro de dependencias con uv. Los originales quedan como respaldo local.
- El usuario autorizó crear el repositorio y eligió expresamente su visibilidad pública: JARO-Hub/wine-quality-machine-learning.
