```bach
src/procesamiento/capa1/
├── __init__.py
├── cli.py                 # Punto de entrada principal con Click
├── config_dicts.py        # DICCIONARIOS: Constantes, columnas esperadas, variables permitidas
├── core_io.py             # MOTOR: Lógica de lectura/escritura en batches con pyarrow
├── rules_yellow.py        # REGLAS: Función de limpieza específica para Yellow
├── rules_green.py         # REGLAS: Función de limpieza específica para Green
└── rules_fhvhv.py         # REGLAS: Función de limpieza específica para HVFHV
```