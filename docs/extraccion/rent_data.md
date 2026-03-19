# Extracción de alquileres NYC

La extracción de alquileres usa ahora como fuente por defecto la **ACS 5-year del
U.S. Census** a nivel de **census tract** para Nueva York, porque los exports
recientes de Inside Airbnb ya no publican precios utilizables en `listings.csv` ni en
`calendar.csv.gz`.

El extractor por defecto construye un parquet con:

- `id` y `zone_id` del tracto
- `borough`
- `neighborhood` y `zone_name`
- `latitude` y `longitude` del centroid del tracto
- `price` como `median gross rent`
- `price_moe` como margen de error ACS

Inside Airbnb sigue disponible como opción manual, pero ya no es la ruta por defecto
del pipeline.

Comandos útiles:

```bash
uv run -m src.extraccion.download_rent_data
uv run -m src.extraccion.download_rent_data --provider acs
uv run -m src.extraccion.download_rent_data --force
uv run -m src.extraccion.download_rent_data --provider insideairbnb --dataset-kind detailed
```
