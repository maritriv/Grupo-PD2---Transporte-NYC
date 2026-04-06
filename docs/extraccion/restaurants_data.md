# Extracción de restaurantes NYC

La extracción de restaurantes usa el dataset oficial
`DOHMH New York City Restaurant Inspection Results` (`43nn-pn8j`) de NYC OpenData.

El cambio importante es que el raw ahora conserva la granularidad real de la fuente:
una inspección puede aparecer en varias filas porque el dataset se publica a nivel de
violación. Por eso el extractor guarda también columnas como:

- `inspection_type`
- `action`
- `violation_code`
- `violation_description`
- `critical_flag`
- `grade_date`

Con esto evitamos los duplicados artificiales que aparecían cuando se tiraban esas
columnas y varias violaciones distintas quedaban reducidas a la misma fila.

El extractor además:

- normaliza `boro`
- convierte fechas y numéricos a tipos consistentes
- elimina solo duplicados exactos reales
- permite sobrescribir con `--force`

Comandos útiles:

```bash
uv run -m src.extraccion.download_restaurants_data --start-year 2024 --end-year 2024
uv run -m src.extraccion.download_restaurants_data --start-year 2024 --end-year 2024 --start-month 1 --end-month 1 --force
uv run -m src.extraccion.main --mode restaurants
```
