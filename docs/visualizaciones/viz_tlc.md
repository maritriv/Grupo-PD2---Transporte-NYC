# Análisis del Mercado de Transporte de Pago en Nueva York (2024)

## Reto del Proyecto

Diseñar una aplicación innovadora, basada en datos, que mejore de forma medible algún aspecto del negocio del transporte de pago en Nueva York.

### Criterios de Evaluación

- Calidad del análisis de mercado
- Uso inteligente de fuentes de datos reales
- Solidez técnica de la propuesta
- Capacidad de diferenciación frente a soluciones existentes

---

## Ejercicio 1: Análisis del Sistema y Visualizaciones

### 1.1 Panorama General del Mercado

![Número de viajes diarios por servicio](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.42%20(3).jpeg)

**Análisis:**
El mercado está claramente dominado por las plataformas VTC (fhvhv - servicios como Uber/Lyft) con aproximadamente 500,000-800,000 viajes diarios, mientras que los taxis tradicionales (yellow) mantienen apenas 80,000-100,000 viajes diarios. Los servicios green tienen presencia casi nula.

**Hallazgos clave:**

- **Ratio de dominio:** Las VTC representan ~85-90% del mercado total
- **Estabilidad temporal:** Patrones relativamente constantes durante el periodo analizado
- **Brecha competitiva:** Los taxis tradicionales han perdido masivamente cuota de mercado

---

### 1.2 Análisis de Demanda por Zonas

#### Zona 132 - Alta demanda

![Demanda por hora en zona 132](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.40.jpeg)

**Análisis:**

- **Peak matutino:** Demanda extremadamente alta a las 0h (~600 viajes VTC) que cae drásticamente hasta las 4h
- **Valle diurno:** Mínimo entre 3-5h (50-100 viajes)
- **Recuperación gradual:** Incremento sostenido desde las 6h hasta las 22h
- **Peak nocturno:** Máximo absoluto a las 22h (~850 viajes VTC)

**Patrón identificado:** Zona de ocio nocturno con fuerte actividad de madrugada y tarde-noche.

---

#### Zona 138 - Alta demanda

![Demanda por hora en zona 138](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.41.jpeg)

**Análisis:**

- **Patrón laboral claro:** Mínima actividad nocturna (0-6h: ~50-100 viajes)
- **Explosión matutina:** Crecimiento dramático desde las 8h (~300 viajes) hasta las 10h (~680 viajes)
- **Mantenimiento alto:** Demanda sostenida entre 600-900 viajes desde las 10h hasta las 23h
- **Peak tarde:** Máximo absoluto a las 19h (~900 viajes VTC)

**Patrón identificado:** Zona de negocios/oficinas con commuting intenso en horas punta.

---

#### Zona 79 - Demanda moderada

![Demanda por hora en zona 79](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.44.jpeg)

**Análisis:**

- **Peak nocturno inicial:** Alta demanda a las 0h (~600 viajes VTC) 
- **Caída pronunciada:** Descenso hasta ~120 viajes a las 4h
- **Recuperación vespertina:** Crecimiento gradual desde las 15h
- **Peak nocturno:** Máximo a las 22h (~710 viajes VTC)

**Patrón identificado:** Zona residencial/ocio con actividad concentrada en tarde-noche.

---

### 1.3 Análisis de Precios

#### Zona 79 - Precios medios

![Precio por hora en zona 79](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.41%20(1).jpeg)

**Análisis:**

- **Surge pricing nocturno VTC:** Pico dramático a las 5h (~€40) por escasez de conductores
- **Precios valle:** Mínimo a las 10h (~€26.50 VTC)
- **Estabilidad diurna:** Precios relativamente constantes €27-32 entre 11h-23h
- **Brecha VTC-Taxi:** VTC cobra €5-10 más que taxis en la mayoría de horas
- **Convergencia parcial:** Los precios se igualan en horas punta (16h)

**Insight crítico:** El surge pricing de VTC puede llegar a +50% en horas de baja oferta.

---

#### Zona 132 - Precios medios

![Precio por hora en zona 132](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.41%20(2).jpeg)

**Análisis:**

- **Precios más altos:** Rango €40-90 vs €68-90 (significativamente superiores a zona 79)
- **Valle nocturno VTC:** Mínimo absoluto a las 4h (~€40) coincide con mínima demanda
- **Peak pricing:** Máximo a las 15-16h (~€90 ambos servicios)
- **Convergencia sostenida:** VTC y taxis mantienen precios similares en horas punta (14-17h)
- **Divergencia nocturna:** VTC más barato que taxis en madrugada (0-5h)

**Insight crítico:** Zona premium con precios 2-3x superiores a otras áreas.

---

#### Zona 138 - Precios medios

![Precio por hora en zona 138](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.42.jpeg)

**Análisis:**

- **Valle matutino:** Precios mínimos a las 4h (~€48-50)
- **Escalada diurna:** Incremento sostenido hasta las 8h (€73 taxis)
- **Convergencia temporal:** Precios similares entre servicios a las 16h (~€69)
- **Estabilidad tarde:** Rango €60-70 desde las 10h hasta las 22h
- **Brecha favorable taxi:** Taxis más caros que VTC en horas valle

**Insight crítico:** Menor volatilidad de precios comparado con otras zonas.

---

### 1.4 Tensión Precio-Volumen

![Tensión: volumen vs variabilidad](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.42%20(1).jpeg)

**Análisis:**

- **Cluster VTC:** Alta concentración en bajo volumen (<25,000 viajes) y baja variabilidad (IQR 10-20)
- **Dispersión VTC:** Algunos outliers con alto volumen (40,000-50,000 viajes) mantienen baja variabilidad
- **Cluster Taxi:** Concentración masiva en bajo volumen (<20,000) y variabilidad muy baja (IQR 8-12)
- **Estabilidad taxi:** Menor dispersión de precios indica pricing más predecible

**Hallazgos clave:**

- VTC tiene mayor variabilidad de precios (surge pricing)
- A mayor volumen, menor es la variabilidad en VTC (economías de escala)
- Taxis mantienen precios más estables independientemente del volumen

---

### 1.5 Estabilidad de Precios Temporal

![Precio medio diario por servicio](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.43.jpeg)

**Análisis:**

- **Convergencia VTC-Taxi:** Precios prácticamente idénticos durante todo el periodo (€25-30)
- **Spike final dramático:** Yellow taxi dispara precios a ~€45 el último día (posible anomalía)
- **Estabilidad VTC:** Fluctuación contenida €26-30 durante 2 meses
- **Green cab:** Consistentemente más barato (€22-23), pero con volumen insignificante

**Insight crítico:** 

- El mercado muestra equilibrio de precios a largo plazo
- El spike final de taxis sugiere evento especial o fallo en datos

---

### 1.6 Oportunidades de Negocio Identificadas

![Top 15 oportunidades](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.42%20(2).jpeg)

**Análisis del Business Score (variabilidad × log(1+volumen)):**

**Top 5 oportunidades:**

1. **265-h18-yellow** (score: ~680): Zona 265, hora 18h, taxis
2. **265-h13-yellow** (score: ~650): Zona 265, hora 13h, taxis
3. **265-h12-yellow** (score: ~630): Zona 265, hora 12h, taxis
4. **230-h6-fhvhv** (score: ~620): Zona 230, hora 6h, VTC
5. **265-h10-yellow** (score: ~610): Zona 265, hora 10h, taxis

**Hallazgos clave:**

- **Dominio zona 265:** 7 de los 15 mejores slots son en zona 265 (yellow)
- **Dominio taxis:** 10 de 15 oportunidades son para taxis tradicionales
- **Zonas estratégicas VTC:** 230 y 132 aparecen múltiples veces para VTC
- **Horas críticas:** Concentración en horas laborales (6h, 10h, 12h, 13h, 14h, 15h, 16h, 18h)

**Interpretación:**
Alta variabilidad + volumen moderado-alto = oportunidad de optimización de pricing y matching.

---

### 1.7 Hotspots Espaciotemporales

#### Hotspots de Demanda

![Hotspots de demanda](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.43%20(1).jpeg)

**Análisis:**

- **Zonas calientes identificadas:** ~25, ~35, ~45 (filas ordenadas por volumen)
- **Peak horario universal:** 8-10h y 17-20h muestran mayor actividad en todas las zonas top
- **Madrugada activa:** Zonas ~25 y ~45 mantienen demanda fuerte 0-2h
- **Valle nocturno:** 3-6h es el periodo más inactivo en todas las zonas

**Patrón espaciotemporal:**

- Las zonas top mantienen actividad todo el día
- Concentración de demanda en horarios laborales típicos
- Segmentación clara entre zonas de ocio nocturno y zonas laborales

---

#### Hotspots de Precio

![Hotspots de precio](../../outputs/viz_tlc/WhatsApp%20Image%202026-02-11%20at%2010.51.43%20(2).jpeg)

**Análisis:**

- **Zona premium identificada:** ~25 (precios altos todo el día)
- **Peak pricing matutino:** 5-8h en múltiples zonas (escasez de oferta)
- **Convergencia diurna:** Precios más homogéneos 12-20h
- **Diferenciación nocturna:** Mayor variabilidad de precios 0-5h

**Correlación con demanda:**

- Zonas con alta demanda NO necesariamente tienen precios altos
- Pricing dinámico más agresivo en horas de baja oferta
- Zonas premium mantienen precios altos independientemente de la hora

---

## Ejercicio 2: Estudio de Mercado

### 2.1 Segmentación de Clientes Potenciales

Basándonos en los patrones identificados:

#### Segmento 1: Commuters Laborales

- **Zonas:** 138, 230, 143
- **Horarios:** 8-10h (ida), 17-19h (vuelta)
- **Volumen:** Alto (600-900 viajes/hora)
- **Sensibilidad al precio:** Media-Alta
- **Comportamiento:** Predecible, recurrente, planificable

#### Segmento 2: Ocio Nocturno

- **Zonas:** 132, 79, 265
- **Horarios:** 22-2h
- **Volumen:** Muy Alto (700-850 viajes/hora)
- **Sensibilidad al precio:** Baja
- **Comportamiento:** Fines de semana, dispuestos a pagar surge pricing

#### Segmento 3: Madrugada (Aeropuertos/Trabajadores nocturnos)

- **Zonas:** Múltiples
- **Horarios:** 0-6h
- **Volumen:** Bajo-Medio
- **Sensibilidad al precio:** Muy Baja (pagan hasta +50%)
- **Comportamiento:** Inelástico, alta urgencia

#### Segmento 4: Día Regular

- **Zonas:** Todas
- **Horarios:** 11-16h
- **Volumen:** Medio
- **Sensibilidad al precio:** Alta
- **Comportamiento:** Comparadores de precio, flexibles

---

### 2.2 Análisis Competitivo

#### Posicionamiento Actual

**VTC (Uber/Lyft):**

- ✅ Dominio absoluto del mercado (85-90%)
- ✅ Tecnología superior (app, tracking, pagos)
- ✅ Flexibilidad de oferta
- ❌ Surge pricing genera rechazo en usuarios
- ❌ Precios menos predecibles

**Taxis Tradicionales:**

- ✅ Precios más estables
- ✅ Regulación establecida
- ❌ Tecnología obsoleta
- ❌ Pérdida masiva de cuota de mercado
- ❌ Menor disponibilidad

**Brecha de Mercado Identificada:**
Los usuarios quieren la conveniencia de VTC con la predictibilidad de precios de los taxis.

---

## Conclusiones

### Hallazgos Clave del Análisis

1. **Dominio VTC absoluto:** 85-90% del mercado, pero con pricing volátil
2. **Oportunidad en taxis:** Infraestructura existente subexplotada
3. **Segmentación clara:** Commuters vs ocio nocturno tienen necesidades distintas
4. **Surge pricing = punto de dolor:** Variabilidad de hasta +50% genera fricción
5. **Zonas premium identificadas:** 132, 265 soportan precios 2-3x superiores