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

function StressLegend({ primaryColor }) {
  return (
    <div style={{ marginBottom: "14px" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "10px",
          marginBottom: "6px",
        }}
      >
        <span
          style={{
            fontSize: "14px",
            fontWeight: 600,
            color: primaryColor,
          }}
        >
          Estable
        </span>

        <div
          style={{
            flex: 1,
            height: "14px",
            borderRadius: "999px",
            background:
              "linear-gradient(90deg, #6cc36a 0%, #b7cf5f 20%, #e0a63a 50%, #d97a45 75%, #d4554a 100%)",
            border: "1px solid #d1d5db",
          }}
        />

        <span
          style={{
            fontSize: "14px",
            fontWeight: 600,
            color: primaryColor,
          }}
        >
          Crítico
        </span>
      </div>

      <div style={{ fontSize: "12px", color: "#6b7280" }}>
        Escala sintética del nivel de tensión urbana
      </div>
    </div>
  )
}

export default function MapView({ zones, primaryColor }) {
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
          throw new Error("El archivo no es un GeoJSON válido de tipo FeatureCollection")
        }

        setDebugInfo(`GeoJSON cargado correctamente: ${data.features.length} zonas`)
        setGeojsonData(data)
      } catch (error) {
        console.error("Error cargando GeoJSON:", error)
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

    return {
      fillColor: getColorFromScore(score),
      weight: 0.8,
      opacity: 1,
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

    try {
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
    } catch (error) {
      console.error("Error creando tooltip:", error)
    }
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
      <h2 style={{ marginTop: 0, color: primaryColor }}>
        Mapa de Estrés Urbano
      </h2>

      <p style={{ color: "#6b7280", marginBottom: "16px" }}>
        Visualización geográfica del nivel de tensión por zona
      </p>

      <StressLegend primaryColor={primaryColor} />

      <div
        style={{
          height: "520px",
          borderRadius: "14px",
          overflow: "hidden",
          border: "1px solid #e5e7eb",
          background: "#f9fafb",
        }}
      >
        {geojsonError ? (
          <div
            style={{
              height: "100%",
              display: "grid",
              placeItems: "center",
              color: "#dc2626",
              padding: "20px",
              textAlign: "center",
            }}
          >
            <div>
              <div>{geojsonError}</div>
              <div style={{ marginTop: "8px", fontSize: "12px", color: "#6b7280" }}>
                {debugInfo}
              </div>
            </div>
          </div>
        ) : geojsonData ? (
          <MapContainer
            center={[40.7128, -74.006]}
            zoom={10}
            style={{ height: "100%", width: "100%" }}
            zoomControl={false}
          >
            <ZoomControl position="topright" />

            <TileLayer
              attribution='&copy; OpenStreetMap contributors &copy; CARTO'
              url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
            />

            <GeoJSON
              key="nyc-zones-layer"
              data={geojsonData}
              style={styleFeature}
              onEachFeature={onEachFeature}
            />
          </MapContainer>
        ) : (
          <div
            style={{
              height: "100%",
              display: "grid",
              placeItems: "center",
              color: "#6b7280",
              textAlign: "center",
              padding: "20px",
            }}
          >
            {debugInfo || "Cargando mapa…"}
          </div>
        )}
      </div>
    </div>
  )
}