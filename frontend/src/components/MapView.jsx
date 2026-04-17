import { useEffect, useMemo, useState } from "react"
import { MapContainer, TileLayer, GeoJSON, ZoomControl } from "react-leaflet"

function getZoneIdFromFeature(feature) {
  const props = feature?.properties || {}

  return Number(
    props.zone_id ??
      props.ZoneID ??
      props.location_id ??
      props.LocationID ??
      props.OBJECTID ??
      props.objectid ??
      props.id
  )
}

function getColorFromScore(score) {
  if (score === null || score === undefined || Number.isNaN(score)) return "#d1d5db"
  if (score < 0.33) return "#6cc36a"
  if (score < 0.66) return "#e0a63a"
  return "#d4554a"
}

function getLevelLabel(score) {
  if (score === null || score === undefined || Number.isNaN(score)) return "sin datos"
  if (score < 0.33) return "low"
  if (score < 0.66) return "medium"
  return "high"
}

function formatDateTime(date) {
  return new Intl.DateTimeFormat("es-ES", {
    weekday: "long",
    day: "numeric",
    month: "long",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date)
}

/* ===================== LEGEND ===================== */

function StressLegend({ primaryColor }) {
  return (
    <div style={{ marginBottom: "14px" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto 1fr auto",
          alignItems: "center",
          gap: "12px",
          marginBottom: "6px",
        }}
      >
        <span style={{ fontSize: "14px", fontWeight: 600, color: primaryColor }}>
          Estable
        </span>

        <div
          style={{
            width: "100%",
            height: "14px",
            borderRadius: "999px",
            background:
              "linear-gradient(90deg, #6cc36a 0%, #b7cf5f 20%, #e0a63a 50%, #d97a45 75%, #d4554a 100%)",
            border: "1px solid #d1d5db",
          }}
        />

        <span style={{ fontSize: "14px", fontWeight: 600, color: primaryColor }}>
          Crítico
        </span>
      </div>

      <div style={{ fontSize: "12px", color: "#6b7280" }}>
        Escala sintética del nivel de tensión urbana
      </div>
    </div>
  )
}

/* ===================== CONTROLS ===================== */

function PredictionControls({
  baseDate,
  targetDate,
  horizonHours,
  setHorizonHours,
  primaryColor,
}) {
  return (
    <div
      style={{
        minWidth: "320px", // 🔥 más ancho el bloque
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        alignItems: "flex-end",
      }}
    >
      <div
        style={{
          fontSize: "13px",
          color: "#6b7280",
          textAlign: "right",
          lineHeight: 1.35,
        }}
      >
        <div>
          <span style={{ fontWeight: 700, color: primaryColor }}>Base:</span>{" "}
          {formatDateTime(baseDate)}
        </div>
        <div>
          <span style={{ fontWeight: 700, color: primaryColor }}>
            Predicción para:
          </span>{" "}
          {formatDateTime(targetDate)}
        </div>
      </div>

      <select
        value={horizonHours}
        onChange={(e) => setHorizonHours(Number(e.target.value))}
        style={{
          width: "220px",        // 🔥 más largo
          padding: "4px 10px",   // 🔥 más fino
          borderRadius: "8px",
          border: "1px solid #d1d5db",
          fontSize: "13px",
          background: "#f9fafb",
          color: "#374151",
          cursor: "pointer",
          alignSelf: "flex-end", // 🔥 alineado con "Base"
        }}
      >
        <option value={0}>Ahora</option>
        {Array.from({ length: 24 }, (_, i) => i + 1).map((h) => (
          <option key={h} value={h}>
            +{h}h
          </option>
        ))}
      </select>
    </div>
  )
}

/* ===================== MAP ===================== */

export default function MapView({
  zones,
  primaryColor,
  baseDate,
  targetDate,
  horizonHours,
  setHorizonHours,
}) {
  const [geojsonData, setGeojsonData] = useState(null)
  const [geojsonError, setGeojsonError] = useState("")
  const [debugInfo, setDebugInfo] = useState("")

  useEffect(() => {
    async function loadGeojson() {
      try {
        setGeojsonError("")
        setDebugInfo("Cargando archivo GeoJSON...")

        const res = await fetch("/nyc-zones.geojson")

        if (!res.ok) {
          throw new Error(`No se pudo cargar el archivo GeoJSON (${res.status})`)
        }

        const data = await res.json()

        if (!data || data.type !== "FeatureCollection" || !Array.isArray(data.features)) {
          throw new Error("El archivo no es un GeoJSON válido")
        }

        setDebugInfo(`GeoJSON cargado: ${data.features.length} zonas`)
        setGeojsonData(data)
      } catch (error) {
        console.error(error)
        setGeojsonError("No se pudo cargar el mapa geográfico")
        setDebugInfo(String(error))
      }
    }

    loadGeojson()
  }, [])

  const scoreByZone = useMemo(() => {
    const map = new Map()
    zones.forEach((z) => {
      map.set(Number(z.zone_id), Number(z.score))
    })
    return map
  }, [zones])

  function styleFeature(feature) {
    const zoneId = getZoneIdFromFeature(feature)
    const score = scoreByZone.get(zoneId)

    if (score === undefined) {
      return {
        fillColor: "#e5e7eb",
        weight: 0.7,
        color: "#ffffff",
        fillOpacity: 0.28,
      }
    }

    return {
      fillColor: getColorFromScore(score),
      weight: 0.8,
      color: "#ffffff",
      fillOpacity: 0.72,
    }
  }

  function onEachFeature(feature, layer) {
    const props = feature?.properties || {}
    const zoneId = getZoneIdFromFeature(feature)
    const score = scoreByZone.get(zoneId)
    const level = getLevelLabel(score)
    const zoneName = props.zone || props.Zone || `Zona ${zoneId ?? "-"}`

    layer.bindTooltip(
      `
        <div style="min-width:140px">
          <strong>${zoneName}</strong><br/>
          ID zona: ${zoneId ?? "-"}<br/>
          Score: ${score !== undefined ? Number(score).toFixed(2) : "sin datos"}<br/>
          Nivel: ${level}
        </div>
      `,
      { sticky: true }
    )
  }

  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "16px",
        padding: "20px",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "20px",
          marginBottom: "8px",
        }}
      >
        <div>
          <h2 style={{ marginTop: 0, marginBottom: "10px", color: primaryColor }}>
            Mapa de Estrés Urbano
          </h2>

          <p style={{ color: "#6b7280", margin: 0 }}>
            Visualización geográfica del nivel de tensión por zona
          </p>
        </div>

        <PredictionControls
          baseDate={baseDate}
          targetDate={targetDate}
          horizonHours={horizonHours}
          setHorizonHours={setHorizonHours}
          primaryColor={primaryColor}
        />
      </div>

      <StressLegend primaryColor={primaryColor} />

      <div
        style={{
          height: "520px",
          borderRadius: "14px",
          overflow: "hidden",
          border: "1px solid #e5e7eb",
        }}
      >
        {geojsonError ? (
          <div style={{ color: "red", padding: "20px" }}>{geojsonError}</div>
        ) : geojsonData ? (
          <MapContainer
            center={[40.7128, -74.006]}
            zoom={10}
            style={{ height: "100%", width: "100%" }}
            zoomControl={false}
          >
            <ZoomControl position="topright" />

            <TileLayer url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" />

            <GeoJSON
              data={geojsonData}
              style={styleFeature}
              onEachFeature={onEachFeature}
            />
          </MapContainer>
        ) : (
          <div style={{ padding: "20px" }}>Cargando mapa…</div>
        )}
      </div>
    </div>
  )
}