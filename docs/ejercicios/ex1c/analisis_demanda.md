# Ejercicio 1(c) - Patrones de Demanda Espacio-Temporales en NYC

## 1. Introducción y Metodología de Procesamiento

El objetivo de este ejercicio es identificar y clasificar los patrones de demanda de taxis en Nueva York, analizando cómo varía el volumen de viajes en función de la **zona de la ciudad** y la **franja horaria**.

### ¿Cómo hemos preparado los datos y por qué?

Para estructurar este análisis, los datos históricos han sido procesados y agregados bajo tres pilares fundamentales:

1. **Agregación a nivel Zona-Hora (`pu_location_id` + `hour`):** En lugar de analizar viajes individuales o días aislados, hemos promediado la actividad a lo largo de todo el histórico para cada combinación de zona y hora. Esto nos permite mitigar el ruido estadístico de anomalías diarias (un día de lluvia extrema, un evento puntual) y nos permite extraer el comportamiento "típico" y recurrente de la ciudad.

2. **Clasificación Relativa de la Demanda (Terciles por Zona):**
   Hemos clasificado el volumen medio de viajes (`num_trips_avg`) en niveles de demanda (**Baja, Media, Alta**) calculando los terciles *de manera independiente para cada zona*.
   
   Si usáramos umbrales globales absolutos, zonas centrales como Manhattan acapararían siempre la demanda "Alta", invisibilizando los picos de actividad en zonas periféricas como Queens o Brooklyn. Al usar terciles por zona, evaluamos el rendimiento de cada barrio respecto a sí mismo, permitiéndonos encontrar el "momento punta" de cualquier rincón de la ciudad.

3. **Medida de Estabilidad (Coeficiente de Variación - CV):**
   Hemos introducido el Coeficiente de Variación ($CV = \frac{\sigma}{\mu}$) utilizando la desviación típica (`num_trips_std`). En base a este CV global, clasificamos la zona-hora como **Predecible**, **Variable** o **Volátil**.
   
   A nivel de negocio o toma de decisiones para un taxista, una demanda "Alta pero Volátil" representa un riesgo mayor que una demanda "Media pero Predecible". Esta métrica añade una capa de fiabilidad crucial para la operativa real.

El resultado es un dataset de 6.000 combinaciones zona-hora perfectamente balanceado (con una distribución casi idéntica de niveles de demanda Baja, Media y Alta).

### Carga de Datos

    Dataset cargado: 6000 filas.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hour</th>
      <th>num_trips_avg</th>
      <th>num_trips_std</th>
      <th>num_trips_min</th>
      <th>num_trips_max</th>
      <th>num_trips_count</th>
      <th>avg_price_mean</th>
      <th>std_price_mean</th>
      <th>demand_level</th>
      <th>stability</th>
      <th>cv</th>
      <th>operational_priority</th>
      <th>operational_priority_label</th>
      <th>pu_location_id</th>
      <th>Borough</th>
      <th>Zone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>95.730839</td>
      <td>43.805106</td>
      <td>38</td>
      <td>320</td>
      <td>1096</td>
      <td>19.506505</td>
      <td>12.836568</td>
      <td>Baja</td>
      <td>Volátil</td>
      <td>0.457586</td>
      <td>1</td>
      <td>Baja</td>
      <td>10</td>
      <td>Queens</td>
      <td>Baisley Park</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>68.035509</td>
      <td>36.584954</td>
      <td>30</td>
      <td>254</td>
      <td>1042</td>
      <td>19.865769</td>
      <td>12.271734</td>
      <td>Baja</td>
      <td>Volátil</td>
      <td>0.537733</td>
      <td>1</td>
      <td>Baja</td>
      <td>10</td>
      <td>Queens</td>
      <td>Baisley Park</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>56.309601</td>
      <td>27.875342</td>
      <td>30</td>
      <td>214</td>
      <td>927</td>
      <td>19.421821</td>
      <td>10.641729</td>
      <td>Baja</td>
      <td>Volátil</td>
      <td>0.495037</td>
      <td>1</td>
      <td>Baja</td>
      <td>10</td>
      <td>Queens</td>
      <td>Baisley Park</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>62.934743</td>
      <td>21.565722</td>
      <td>30</td>
      <td>210</td>
      <td>1088</td>
      <td>19.666065</td>
      <td>9.535722</td>
      <td>Baja</td>
      <td>Volátil</td>
      <td>0.342668</td>
      <td>1</td>
      <td>Baja</td>
      <td>10</td>
      <td>Queens</td>
      <td>Baisley Park</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>91.666971</td>
      <td>19.632036</td>
      <td>42</td>
      <td>174</td>
      <td>1096</td>
      <td>20.081418</td>
      <td>9.602911</td>
      <td>Baja</td>
      <td>Predecible</td>
      <td>0.214167</td>
      <td>3</td>
      <td>Baja</td>
      <td>10</td>
      <td>Queens</td>
      <td>Baisley Park</td>
    </tr>
  </tbody>
</table>
</div>


## Análisis descriptivo

### Insights del Análisis descriptivo

**1. Validación de la Demanda (Terciles Equilibrados)**
Como esperábamos por diseño, la distribución de la demanda es perfectamente equitativa (Baja: **33.5%**, Media: **33.2%**, Alta: **33.3%**). Esto confirma el éxito de nuestra estrategia de cálculo por terciles a nivel de zona. Al no usar valores absolutos, evitamos el sesgo que concentraría toda la "demanda alta" en Manhattan, logrando identificar los momentos de máxima actividad relativos a cada barrio (por ejemplo, el "pico" de actividad de un barrio periférico de Queens).

**2. La Dualidad de la Estabilidad en NYC**
Curiosamente, la variabilidad del servicio se divide de forma casi idéntica en tercios (Volátil: **34.0%**, Variable: **33.0%**, Predecible: **33.0%**). Esto revela una dinámica de movilidad dual en Nueva York: 
* Un tercio de las combinaciones zona-hora responde a rutinas férreas (probablemente desplazamientos laborales en días laborables).
* Otro tercio representa un caos operativo, donde el volumen de viajes fluctúa radicalmente (muy probablemente asociado a eventos, ocio nocturno en fines de semana o cambios meteorológicos).

**3. Foco Operativo y Rentabilidad**
La distribución de la Prioridad Operativa es la métrica más reveladora para el modelo de negocio. Prácticamente la mitad de las situaciones (**49.7%**) se clasifican como de prioridad Baja. Por el contrario, tenemos un **38.2%** de prioridad Alta y un segmento muy específico del **12.2%** de prioridad Crítica.


    
![png](analisis_demanda_files/analisis_demanda_6_0.png)
    


    --- PROPORCIONES EXACTAS (%) ---
               Demanda (%) Estabilidad (%) Prioridad (%)
    Baja              33.5               -          49.7
    Alta              33.3               -          38.2
    Media             33.2               -             -
    Volátil              -            34.0             -
    Predecible           -            33.0             -
    Variable             -            33.0             -
    Crítica              -               -          12.2


### Evolución Temporal

**1. El Valle Operativo de Madrugada (2:00 - 5:00)**
Los datos demuestran un colapso natural de la movilidad durante la madrugada, tocando su mínimo absoluto a las 4:00 AM con apenas **91.2** viajes de media. El mapa de calor refleja esto claramente, mostrando un bloque sólido de demanda "Baja" en esas horas. 
* *Implicación:* Esta es la ventana valle (sin coste de oportunidad) ideal para que las flotas programen mantenimientos, repostajes/recargas de vehículos o cambios de turno de los conductores.

**2. El "Golden Peak" de la Tarde/Noche (17:00 - 20:00)**
La ciudad experimenta su máxima tensión de movilidad en el bloque de tarde, cruzando el final de la jornada laboral con el inicio de la actividad de ocio/cenas. El pico absoluto se alcanza a las 18:00 (**357.6** viajes de media, casi un 400% más que en la madrugada). En el mapa de calor, la etiqueta de demanda "Alta" domina por completo desde las 15:00 hasta las 20:00.
* *Implicación:* Este es el momento crítico de facturación. La incapacidad de absorber esta demanda genera frustración en el usuario y pérdida de ingresos.

**3. La Estabilidad del Mediodía (10:00 - 14:00)**
A diferencia del pico matutino (8:00 - 9:00) y el gran pico vespertino, las horas centrales del día presentan un volumen de viajes estable y constante. El mapa de calor muestra una alta concentración de etiquetas de demanda "Media". Es un periodo de transición altamente predecible.


    
![png](analisis_demanda_files/analisis_demanda_8_0.png)
    


    --- HORAS VALLE (Menor volumen medio de viajes) ---
    hour
    4     91.2
    3     98.6
    5    102.1
    2    122.5
    Name: num_trips_avg, dtype: Float64
    
    --- HORAS PICO (Mayor volumen medio de viajes) ---
    hour
    18    357.6
    19    344.3
    17    341.0
    20    324.2
    Name: num_trips_avg, dtype: Float64


### Patrones por zona (heterogeneidad espacial)

**1. El monopolio de los Aeropuertos (Nodos de Transporte)**
JFK y LaGuardia lideran de forma aplastante el volumen de la ciudad, superando los **1.300 viajes medios**. Como observamos en su evolución diaria, tienen un ecosistema propio: su demanda no obedece al típico horario de oficina, sino que crece de forma sostenida a lo largo del día y se mantiene muy alta hasta bien entrada la noche, guiada por la programación de vuelos.
* *Implicación:* Los aeropuertos son "agujeros negros" de demanda. Requieren una flota cautiva o un sistema de asignación de colas independiente al del resto de la ciudad.

**2. El latido de los Negocios (Midtown Center)**
La zona de *Midtown Center* ocupa el tercer lugar (1.029 viajes medios), pero su comportamiento es el clásico de un hub financiero/comercial. Su curva muestra un despertar agresivo por la mañana, un valle al mediodía y un pico explosivo por la tarde (cuando las oficinas cierran), para desplomarse rápidamente durante la noche cuando el barrio se vacía.

**3. El relevo del Ocio y lo Residencial**
El resto del Top 10 nos confirma cómo se mueve el dinero en la ciudad tras la jornada laboral. Zonas puramente de ocio y restauración nocturna (*Times Sq/Theatre District*, *East Village*, *East Chelsea*) conviven en volumen con los grandes bloques residenciales (*Upper East Side South/North*). Esto demuestra una migración pendular: los usuarios viajan de las zonas de negocio a las de ocio, y finalmente a las residenciales.


    
![png](analisis_demanda_files/analisis_demanda_10_0.png)
    


    --- TOP 10 ZONAS POR VOLUMEN MEDIO ---
    Zone
    JFK Airport                  1362.9
    LaGuardia Airport            1331.7
    Midtown Center               1029.5
    Times Sq/Theatre District     923.1
    East Village                  900.8
    Upper East Side South         840.3
    East Chelsea                  812.6
    Union Sq                      798.7
    Midtown South                 749.2
    Upper East Side North         742.9
    Name: num_trips_avg, dtype: Float64


### Demanda vs Estabilidad

Al cruzar el Nivel de Demanda con nuestro índice de Estabilidad (basado en el CV), descubrimos la verdadera anatomía del riesgo operativo en Nueva York.

**1. El "Core" del Negocio (Alta Demanda + Predecible)**
Existen **730 combinaciones** de zona-hora que son de demanda Alta y comportamiento Predecible. Estos son nuestros **"Hotspots Fiables"**. 
* *Implicación:* Son el motor económico base de la ciudad (rutinas de oficina, nudos de transporte principales).

**2. El Gran Reto Algorítmico (Alta Demanda + Volátil)**
El dato más crítico de la matriz son las **544 combinaciones** de demanda Alta pero Volátil. Son momentos donde la ciudad "estalla" en demanda, pero no de forma consistente todos los días (eventos deportivos, tormentas imprevistas, picos de ocio anómalos).
* *Implicación:* Aquí es donde las empresas de la competencia fallan, dejando usuarios sin coche o aplicando tarifas dinámicas abusivas.

**3. El "Ruido" Operativo (Baja Demanda + Volátil)**
La gran mayoría de la volatilidad de la ciudad (**1.068 casos**) ocurre en escenarios de Baja demanda. Esto tiene sentido matemático (un par de viajes extra en una zona vacía disparan la varianza temporal), pero operativamente es una trampa.
* *Implicación:* Enviar conductores a estas zonas guiándose por "picos repentinos" es un error que genera kilómetros en vacío (coste puro). Se pueden considerar como falsos positivos.


    
![png](analisis_demanda_files/analisis_demanda_12_0.png)
    


    --- MATRIZ CRUZADA (DEMANDA VS ESTABILIDAD) ---
    stability     Predecible  Variable  Volátil
    demand_level                               
    Baja                 481       459     1068
    Media                769       798      428
    Alta                 730       723      544


### Prioridad operacional

La etiqueta de Prioridad Operacional traduce todo el análisis estadístico previo a una métrica de decisión directa para la plataforma. 

**1. Dimensionando el "Terreno de Juego"**
De todas las combinaciones posibles de zona-hora en la ciudad, solo **730 casos (12.2%)** caen en la categoría "Crítica" (Alta demanda + Alta volatilidad), frente a los **2.980 casos** de prioridad Baja. Esto demuestra que la ciudad no es un caos constante; el problema está altamente concentrado.

**2. El Desfase entre "Hora Pico" y "Hora Crítica"**
Este es el descubrimiento más importante del análisis descriptivo. Anteriormente observamos que el volumen máximo de viajes puros ocurre a las 18:00 y 19:00. Sin embargo, la **máxima criticidad operativa se adelanta a las 17:00 (122 zonas críticas)**, con un bloque altamente inestable que se gesta desde las 14:00 (98 zonas críticas). 
* *Implicación:* La transición de la tarde (salidas de colegios, fin de jornadas tempranas, recados de última hora) genera escenarios mucho más impredecibles que la hora punta de la noche, que aunque masiva, es más estable y rutinaria.

**Conclusión Final de Negocio:**
Un sistema de gestión de flotas tradicional tiende a basarse en promedios históricos, lo que suele llevar a concentrar la oferta en torno a las 19:00. Sin embargo, para ese momento la oportunidad ya está parcialmente perdida. Un enfoque basado en datos permite identificar que la ventana clave de rentabilidad se sitúa entre las 15:00 y las 17:00. Anticiparse a ese periodo, posicionando la flota en las zonas con mayor volatilidad antes de que aumente la demanda, contribuye a reducir los tiempos de espera del usuario y a maximizar el tiempo efectivo de servicio de los conductores.


    
![png](analisis_demanda_files/analisis_demanda_14_0.png)
    


    --- DISTRIBUCIÓN DE PRIORIDAD ---
    operational_priority_label
    Baja       2980
    Alta       2290
    Crítica     730
    Name: count, dtype: int64
    
    --- TOP 5 HORAS CON MÁS ZONAS CRÍTICAS ---
    hour
    17    122
    15    114
    16    107
    14     98
    18     80
    Name: Crítica, dtype: int64


## clustering sobre comportamiento temporal

Hasta ahora hemos analizado la ciudad como un todo o por zonas individuales. Sin embargo, para una gestión de flota eficiente, no podemos tratar a las 263 zonas de forma aislada. El objetivo de este clustering es identificar "Arquetipos de Barrios".

Al normalizar los datos, eliminamos la diferencia de volumen (que una zona tenga 1.000 viajes y otra 10) y nos centramos exclusivamente en la forma de su demanda. Queremos responder a: ¿Esta zona despierta por la mañana? ¿Es una zona de tarde? ¿O es un nodo que nunca duerme?


    
![png](analisis_demanda_files/analisis_demanda_16_0.png)
    


    
    --- Cluster 0 (76 zonas) ---



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster</th>
      <th>num_trips_avg</th>
    </tr>
    <tr>
      <th>Zone</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>JFK Airport</th>
      <td>0</td>
      <td>1362.9</td>
    </tr>
    <tr>
      <th>LaGuardia Airport</th>
      <td>0</td>
      <td>1331.7</td>
    </tr>
    <tr>
      <th>Midtown Center</th>
      <td>0</td>
      <td>1029.5</td>
    </tr>
    <tr>
      <th>Times Sq/Theatre District</th>
      <td>0</td>
      <td>923.1</td>
    </tr>
    <tr>
      <th>East Chelsea</th>
      <td>0</td>
      <td>812.6</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- Cluster 1 (55 zonas) ---



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster</th>
      <th>num_trips_avg</th>
    </tr>
    <tr>
      <th>Zone</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Richmond Hill</th>
      <td>1</td>
      <td>231.1</td>
    </tr>
    <tr>
      <th>Saint Albans</th>
      <td>1</td>
      <td>220.0</td>
    </tr>
    <tr>
      <th>Van Cortlandt Village</th>
      <td>1</td>
      <td>197.1</td>
    </tr>
    <tr>
      <th>Morrisania/Melrose</th>
      <td>1</td>
      <td>193.8</td>
    </tr>
    <tr>
      <th>Baisley Park</th>
      <td>1</td>
      <td>178.0</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- Cluster 2 (99 zonas) ---



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster</th>
      <th>num_trips_avg</th>
    </tr>
    <tr>
      <th>Zone</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Upper East Side South</th>
      <td>2</td>
      <td>840.3</td>
    </tr>
    <tr>
      <th>Upper East Side North</th>
      <td>2</td>
      <td>742.9</td>
    </tr>
    <tr>
      <th>East New York</th>
      <td>2</td>
      <td>577.2</td>
    </tr>
    <tr>
      <th>Upper West Side South</th>
      <td>2</td>
      <td>565.1</td>
    </tr>
    <tr>
      <th>Lenox Hill West</th>
      <td>2</td>
      <td>539.5</td>
    </tr>
  </tbody>
</table>
</div>


    
    --- Cluster 3 (25 zonas) ---



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cluster</th>
      <th>num_trips_avg</th>
    </tr>
    <tr>
      <th>Zone</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>East Village</th>
      <td>3</td>
      <td>900.8</td>
    </tr>
    <tr>
      <th>West Village</th>
      <td>3</td>
      <td>689.1</td>
    </tr>
    <tr>
      <th>Lower East Side</th>
      <td>3</td>
      <td>648.7</td>
    </tr>
    <tr>
      <th>Bushwick South</th>
      <td>3</td>
      <td>579.0</td>
    </tr>
    <tr>
      <th>Williamsburg (North Side)</th>
      <td>3</td>
      <td>543.1</td>
    </tr>
  </tbody>
</table>
</div>


Tras cruzar los clusters con las zonas reales, identificamos los 4 pilares de la movilidad en NYC:

- **Cluster 0: Centros Neurálgicos y Grandes Conectores (76 zonas)**

    - **Zonas representativas**: JFK Airport, LaGuardia Airport, Midtown Center, Times Square.

    - **Perfil**: Es el "corazón" de la actividad. Son zonas de volumen masivo que no descansan. Combinan el flujo constante de los aeropuertos con el bullicio turístico y de negocios de Midtown.

- **Cluster 1: Periferia y Barrios de Baja Densidad (55 zonas)**

    - **Zonas representativas**: Richmond Hill, Saint Albans, Morrisania (Queens y Bronx).

    - **Perfil**: Son zonas con volúmenes mucho más bajos (num_trips_avg entre 170-230). Representan barrios periféricos donde el taxi/VTC se usa de forma más esporádica o para trayectos muy específicos (posiblemente conexiones con el metro).

- **Cluster 2: El Cinturón Residencial y de "Commuters" (99 zonas)**

    - **Zonas representativas**: Upper East Side (North/South), Upper West Side, East New York.

    - **Perfil**: Es el grupo más grande. Representa donde vive la gente que trabaja en el Cluster 0. Su comportamiento es el más predecible: el coche se pide para salir de casa (mañana) y para volver (tarde).

- **Cluster 3: Distritos de Ocio y Vida Nocturna (25 zonas)**

    - **Zonas representativas**: East Village, West Village, Lower East Side, Bushwick South.

    - **Perfil**: Zonas de alta densidad pero con un ritmo "tardío". El volumen es alto (900 viajes en East Village), pero sabemos por su perfil que este volumen explota cuando el resto de la ciudad empieza a dormir.

### El Mapa Estratégico: Geografía de la Oportunidad

El mapa coroplético confirma que el comportamiento de la demanda en Nueva York no es aleatorio, sino que responde a una **lógica urbanística y funcional** perfectamente definida. 

**¿Qué estamos viendo en el mapa?**

1. **La "Isla de la Actividad" (Cluster 0):** Manhattan aparece dominado por el color del Cluster 0, extendiéndose desde el Midtown hacia los dos grandes aeropuertos. Esta "columna vertebral" de la ciudad es el área de máxima facturación y requiere una presencia de flota constante.

2. **El Cinturón de Dormitorios (Cluster 2):** Observamos cómo las zonas residenciales (Upper East/West Side y partes de Brooklyn) rodean los centros de negocio. Este patrón indica que la flota debe "fluir" hacia el centro por la mañana y "replegarse" hacia estos barrios por la tarde.

3. **Enclaves de Ocio (Cluster 3):** Los focos de vida nocturna en el Village y Williamsburg aparecen como "islas" de alta rentabilidad nocturna. 

4. **Zonas de Baja Prioridad (Cluster 1):** La periferia muestra una demanda esparcida y menos estructurada.


    
![png](analisis_demanda_files/analisis_demanda_20_0.png)
    

