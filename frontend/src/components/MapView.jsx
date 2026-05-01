import { useEffect, useMemo, useRef, useState } from "react"
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

function getZoneNameFromFeature(feature) {
  const props = feature?.properties || {}
  const zoneId = getZoneIdFromFeature(feature)

  return props.zone || props.Zone || `Zona ${zoneId ?? "-"}`
}

function getColorFromScore(score) {
  if (score === null || score === undefined || Number.isNaN(score)) return "#d1d5db"
  if (score <= 0.2) return "#6cc36a"
  if (score <= 0.4) return "#a7c957"
  if (score <= 0.6) return "#e0a63a"
  if (score <= 0.8) return "#d97a45"
  return "#d4554a"
}

function getStressLabel(score) {
  if (score === null || score === undefined || Number.isNaN(score)) return "Sin datos"
  if (score <= 0.2) return "Estable"
  if (score <= 0.4) return "Estrés bajo"
  if (score <= 0.6) return "Estrés moderado"
  if (score <= 0.8) return "Estrés alto"
  return "Crítico"
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
              "linear-gradient(90deg, #6cc36a 0%, #a7c957 25%, #e0a63a 50%, #d97a45 75%, #d4554a 100%)",
            border: "1px solid #d1d5db",
          }}
        />

        <span style={{ fontSize: "14px", fontWeight: 600, color: primaryColor }}>
          Crítico
        </span>
      </div>

      <div style={{ fontSize: "12px", color: "#6b7280" }}>
        Índice continuo de estrés urbano entre 0.0 y 1.0
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
        minWidth: "320px",
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
          width: "220px",
          padding: "4px 10px",
          borderRadius: "8px",
          border: "1px solid #d1d5db",
          fontSize: "13px",
          background: "#f9fafb",
          color: "#374151",
          cursor: "pointer",
          alignSelf: "flex-end",
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

/* ===================== SEARCH ===================== */

function ZoneSearch({
  searchTerm,
  setSearchTerm,
  suggestions,
  selectedZoneId,
  clearSearch,
  selectZone,
  primaryColor,
}) {
  return (
    <div style={{ position: "relative", marginBottom: "14px" }}>
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && suggestions.length > 0) {
            selectZone(suggestions[0])
          }
        }}
        placeholder="Buscar zona en el mapa..."
        style={{
          width: "100%",
          boxSizing: "border-box",
          padding: "10px 38px 10px 12px",
          borderRadius: "10px",
          border: "1px solid #d1d5db",
          fontSize: "14px",
          color: "#111827",
          outline: "none",
          background: "white",
        }}
      />

      {searchTerm && (
        <button
          onClick={clearSearch}
          aria-label="Limpiar búsqueda"
          style={{
            position: "absolute",
            right: "10px",
            top: "10px",
            border: "none",
            background: "transparent",
            fontSize: "16px",
            cursor: "pointer",
            color: "#6b7280",
            lineHeight: 1,
          }}
        >
          ✕
        </button>
      )}

      {searchTerm.trim() && suggestions.length > 0 && (
        <div
          style={{
            position: "absolute",
            zIndex: 1000,
            top: "46px",
            left: 0,
            right: 0,
            background: "white",
            border: "1px solid #e5e7eb",
            borderRadius: "10px",
            boxShadow: "0 10px 24px rgba(15,23,42,0.12)",
            overflow: "hidden",
          }}
        >
          {suggestions.map((zone) => (
            <button
              key={zone.id}
              onClick={() => selectZone(zone)}
              style={{
                width: "100%",
                textAlign: "left",
                padding: "10px 12px",
                border: "none",
                borderBottom: "1px solid #f3f4f6",
                background: selectedZoneId === zone.id ? "#f3f4f6" : "white",
                color: primaryColor,
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              {zone.name}
              <span style={{ color: "#6b7280", fontWeight: 400 }}>
                {" "}· ID {zone.id}
              </span>
            </button>
          ))}
        </div>
      )}

      {searchTerm.trim() && suggestions.length === 0 && (
        <div
          style={{
            position: "absolute",
            zIndex: 1000,
            top: "46px",
            left: 0,
            right: 0,
            background: "white",
            border: "1px solid #e5e7eb",
            borderRadius: "10px",
            padding: "10px 12px",
            color: "#6b7280",
            boxShadow: "0 10px 24px rgba(15,23,42,0.12)",
          }}
        >
          No se han encontrado zonas
        </div>
      )}
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
  const [searchTerm, setSearchTerm] = useState("")
  const [selectedZoneId, setSelectedZoneId] = useState(null)

  const mapRef = useRef(null)
  const layerByZoneRef = useRef(new Map())

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

  const zoneOptions = useMemo(() => {
    if (!geojsonData) return []

    return (geojsonData.features || [])
      .map((feature) => {
        const id = getZoneIdFromFeature(feature)
        const name = getZoneNameFromFeature(feature)
        const score = scoreByZone.get(id)

        return {
          id,
          name,
          score,
        }
      })
      .filter((zone) => !Number.isNaN(zone.id))
      .sort((a, b) => a.name.localeCompare(b.name))
  }, [geojsonData, scoreByZone])

  const suggestions = useMemo(() => {
    const normalized = searchTerm.trim().toLowerCase()
    if (!normalized) return []

    return zoneOptions
      .filter((zone) => {
        return (
          zone.name.toLowerCase().includes(normalized) ||
          String(zone.id).includes(normalized)
        )
      })
      .slice(0, 6)
  }, [searchTerm, zoneOptions])

  function styleFeature(feature) {
    const zoneId = getZoneIdFromFeature(feature)
    const score = scoreByZone.get(zoneId)
    const isSelected = selectedZoneId === zoneId

    if (score === undefined) {
      return {
        fillColor: "#e5e7eb",
        weight: isSelected ? 3 : 0.7,
        color: isSelected ? "#111827" : "#ffffff",
        fillOpacity: isSelected ? 0.85 : 0.28,
      }
    }

    return {
      fillColor: getColorFromScore(score),
      weight: isSelected ? 3 : 0.8,
      color: isSelected ? "#111827" : "#ffffff",
      fillOpacity: isSelected ? 0.9 : 0.72,
    }
  }

  function onEachFeature(feature, layer) {
    const zoneId = getZoneIdFromFeature(feature)
    layerByZoneRef.current.set(zoneId, layer)

    const props = feature?.properties || {}
    const score = scoreByZone.get(zoneId)
    const label = getStressLabel(score)
    const zoneName = props.zone || props.Zone || `Zona ${zoneId ?? "-"}`

    layer.bindTooltip(
      `
        <div style="min-width:150px">
          <strong>${zoneName}</strong><br/>
          ID zona: ${zoneId ?? "-"}<br/>
          Índice: ${score !== undefined ? Number(score).toFixed(2) : "sin datos"}<br/>
          Estado: ${label}
        </div>
      `,
      { sticky: true }
    )
  }

  function selectZone(zone) {
    setSelectedZoneId(zone.id)
    setSearchTerm(zone.name)

    const layer = layerByZoneRef.current.get(zone.id)
    const map = mapRef.current

    if (layer && map) {
      map.fitBounds(layer.getBounds(), {
        padding: [30, 30],
        maxZoom: 13,
      })

      setTimeout(() => {
        layer.openTooltip()
      }, 250)
    }
  }

  function clearSearch() {
    setSearchTerm("")
    setSelectedZoneId(null)

    if (mapRef.current) {
      mapRef.current.setView([40.7128, -74.006], 10)
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

      <ZoneSearch
        searchTerm={searchTerm}
        setSearchTerm={setSearchTerm}
        suggestions={suggestions}
        selectedZoneId={selectedZoneId}
        clearSearch={clearSearch}
        selectZone={selectZone}
        primaryColor={primaryColor}
      />

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
            ref={mapRef}
            center={[40.7128, -74.006]}
            zoom={10}
            style={{ height: "100%", width: "100%" }}
            zoomControl={false}
          >
            <ZoomControl position="topright" />

            <TileLayer url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" />

            <GeoJSON
              key={`${selectedZoneId ?? "none"}-${zones.length}`}
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