```
nyc-transport-project/
├─ data/
│  ├─ raw/
│  │  ├─ yellow/      # Parquet originales descargados
│  │  ├─ green/
│  │  ├─ fhv/
│  │  └─ hvfhv/
│  ├─ interim/        # Muestreos, recortes por año/mes
│  └─ processed/      # Tablas agregadas para modelos/gráficos
├─ notebooks/
│  ├─ 01_faseA_exploracion.ipynb
│  ├─ 02_faseB_patrones_tensiones.ipynb
│  ├─ 03_faseC_problemas_candidatos.ipynb
│  ├─ 04_faseD_profundizacion_problema.ipynb
│  ├─ 05_faseE_mercado_monetizacion.ipynb
│  └─ 06_faseF_propuesta_plan.ipynb
├─ src/
│  ├─ data/
│  │  ├─ download_tlc_data.py      # scripts de descarga
│  │  └─ load_schema_utils.py      # funciones de lectura (pyarrow/spark)
│  ├─ features/
│  │  └─ build_features.py         # agregaciones, indicadores, etc.
│  ├─ viz/
│  │  └─ plotting_utils.py         # funciones para mapas, gráficos
│  └─ models/
├─ docs/
│  ├─ entrega1/
│  │  └─ memoria_entrega1.md       # borrador informe
│  └─ referencias.md               # enlaces a TLC, papers, etc.
├─ config/
│  ├─ data_sources.yaml            # qué años/servicios descargamos
├─ tests/
│  └─ test_data_utils.py           # pruebas básicas de funciones
├─ .gitignore
├─ README.md
└─ pyproject.toml
```