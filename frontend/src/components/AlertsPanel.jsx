import { useEffect, useMemo, useState } from "react"
import { getZoneForecast } from "../api/client"

function getStressLabel(score) {
  if (score === null || score === undefined || Number.isNaN(score)) return "Sin datos"
  if (score <= 0.2) return "Estable"
  if (score <= 0.4) return "Bajo estrés"
  if (score <= 0.6) return "Estrés moderado"
  if (score <= 0.8) return "Alto estrés"
  return "Crítico"
}

function getStressColor(score) {
  if (score === null || score === undefined || Number.isNaN(score)) return "#6b7280"
  if (score <= 0.2) return "#6cc36a"
  if (score <= 0.4) return "#84a63a"
  if (score <= 0.6) return "#c98b1f"
  if (score <= 0.8) return "#d97a45"
  return "#dc2626"
}

function ForecastDetails({ zoneId, zoneName, score, primaryColor, dayOfWeek, hour }) {
  const [forecast, setForecast] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    async function loadForecast() {
      try {
        setLoading(true)
        setError("")

        const data = await getZoneForecast(zoneId, dayOfWeek, hour)

        const parsed = (data.forecast || []).map((item) => ({
          timeLabel: item.time_label,
          score: Number(item.score),
          rawStress: Number(item.raw_stress),
          label: getStressLabel(Number(item.score)),
          color: getStressColor(Number(item.score)),
        }))

        setForecast(parsed)
      } catch (err) {
        console.error(err)
        setError("No se pudo cargar la previsión real")
        setForecast([])
      } finally {
        setLoading(false)
      }
    }

    loadForecast()
  }, [zoneId, dayOfWeek, hour])

  if (loading) {
    return <div style={{ marginTop: "12px", color: "#6b7280" }}>Cargando previsión real...</div>
  }

  if (error) {
    return <div style={{ marginTop: "12px", color: "#dc2626" }}>{error}</div>
  }

  if (forecast.length === 0) {
    return <div style={{ marginTop: "12px", color: "#6b7280" }}>Sin previsión disponible</div>
  }

  const peak = forecast.reduce((best, item) => (item.score > best.score ? item : best))
  const average = forecast.reduce((acc, item) => acc + item.score, 0) / forecast.length

  return (
    <div
      style={{
        marginTop: "12px",
        padding: "14px",
        borderRadius: "12px",
        background: "#f9fafb",
        border: "1px solid #e5e7eb",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "10px",
          marginBottom: "14px",
        }}
      >
        <div style={{ padding: "10px", borderRadius: "10px", background: "white", border: "1px solid #e5e7eb" }}>
          <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>Estado actual</div>
          <div style={{ color: getStressColor(score), fontWeight: 700 }}>{getStressLabel(score)}</div>
        </div>

        <div style={{ padding: "10px", borderRadius: "10px", background: "white", border: "1px solid #e5e7eb" }}>
          <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>Score actual</div>
          <div style={{ color: primaryColor, fontWeight: 700 }}>{score.toFixed(2)}</div>
        </div>

        <div style={{ padding: "10px", borderRadius: "10px", background: "white", border: "1px solid #e5e7eb" }}>
          <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>Pico previsto</div>
          <div style={{ color: getStressColor(peak.score), fontWeight: 700 }}>{peak.score.toFixed(2)}</div>
        </div>

        <div style={{ padding: "10px", borderRadius: "10px", background: "white", border: "1px solid #e5e7eb" }}>
          <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>Horizonte de mayor tensión</div>
          <div style={{ color: primaryColor, fontWeight: 700 }}>{peak.timeLabel}</div>
        </div>
      </div>

      <div style={{ marginBottom: "8px", fontWeight: 700, color: primaryColor }}>
        Evolución prevista de {zoneName}
      </div>

      <div
        style={{
          overflow: "hidden",
          borderRadius: "10px",
          border: "1px solid #e5e7eb",
          background: "white",
        }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "13px" }}>
          <thead>
            <tr style={{ background: "#f3f4f6", color: "#374151" }}>
              <th style={{ textAlign: "left", padding: "8px 10px" }}>Horizonte</th>
              <th style={{ textAlign: "left", padding: "8px 10px" }}>Score</th>
              <th style={{ textAlign: "left", padding: "8px 10px" }}>Stress real</th>
              <th style={{ textAlign: "left", padding: "8px 10px" }}>Estado</th>
            </tr>
          </thead>
          <tbody>
            {forecast.map((item) => (
              <tr key={item.timeLabel} style={{ borderTop: "1px solid #e5e7eb" }}>
                <td style={{ padding: "8px 10px" }}>{item.timeLabel}</td>
                <td style={{ padding: "8px 10px" }}>{item.score.toFixed(2)}</td>
                <td style={{ padding: "8px 10px" }}>{item.rawStress.toFixed(2)}</td>
                <td style={{ padding: "8px 10px", color: item.color, fontWeight: 600 }}>{item.label}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: "10px", fontSize: "12px", color: "#6b7280" }}>
        Media prevista próximas horas: <strong>{average.toFixed(2)}</strong>
      </div>
    </div>
  )
}

export default function AlertsPanel({ zones, primaryColor, dayOfWeek, hour }) {
  const [zoneNames, setZoneNames] = useState({})
  const [expandedZoneId, setExpandedZoneId] = useState(null)
  const [sortOrder, setSortOrder] = useState("desc")
  const [searchTerm, setSearchTerm] = useState("")

  useEffect(() => {
    async function loadZoneNames() {
      try {
        const res = await fetch("/nyc-zones.geojson")
        if (!res.ok) throw new Error("No se pudo cargar el GeoJSON")

        const data = await res.json()
        const mapping = {}

        for (const feature of data.features || []) {
          const props = feature.properties || {}
          const id = Number(
            props.LocationID ??
              props.zone_id ??
              props.ZoneID ??
              props.OBJECTID ??
              props.objectid ??
              props.id
          )
          const name = props.zone || props.Zone || `Zona ${id}`

          if (!Number.isNaN(id)) {
            mapping[id] = name
          }
        }

        setZoneNames(mapping)
      } catch (error) {
        console.error("Error cargando nombres de zonas:", error)
      }
    }

    loadZoneNames()
  }, [])

  const alerts = useMemo(() => {
    const normalizedSearch = searchTerm.trim().toLowerCase()

    let filtered = [...zones]

    if (normalizedSearch) {
      filtered = filtered.filter((z) => {
        const zoneId = Number(z.zone_id)
        const zoneName = zoneNames[zoneId] || `Zona ${zoneId}`

        return (
          zoneName.toLowerCase().includes(normalizedSearch) ||
          String(zoneId).includes(normalizedSearch)
        )
      })
    } else {
      filtered = filtered.filter((z) => Number(z.score) >= 0.0)
    }

    return filtered
      .sort((a, b) =>
        sortOrder === "desc"
          ? Number(b.score) - Number(a.score)
          : Number(a.score) - Number(b.score)
      )
      .slice(0, 4)
  }, [zones, zoneNames, searchTerm, sortOrder])

  function toggleDetails(zoneId) {
    setExpandedZoneId((current) => (current === zoneId ? null : zoneId))
  }

  const panelTitle = searchTerm.trim() ? "Resultados" : "Alertas"

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
          alignItems: "center",
          gap: "12px",
          marginBottom: "14px",
        }}
      >
        <h3 style={{ margin: 0, color: primaryColor }}>{panelTitle}</h3>

        <select
          value={sortOrder}
          onChange={(e) => setSortOrder(e.target.value)}
          style={{
            border: "1px solid #d1d5db",
            borderRadius: "8px",
            padding: "6px 8px",
            fontSize: "12px",
            color: primaryColor,
            background: "white",
            cursor: "pointer",
            fontWeight: 600,
          }}
        >
          <option value="desc">Mayor estrés</option>
          <option value="asc">Menor estrés</option>
        </select>
      </div>

      <div style={{ position: "relative", marginBottom: "18px" }}>
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => {
            setSearchTerm(e.target.value)
            setExpandedZoneId(null)
          }}
          placeholder="Buscar zona..."
          style={{
            width: "100%",
            boxSizing: "border-box",
            padding: "10px 36px 10px 12px",
            borderRadius: "10px",
            border: "1px solid #d1d5db",
            fontSize: "14px",
            color: "#111827",
            outline: "none",
          }}
        />

        {searchTerm && (
          <button
            onClick={() => {
              setSearchTerm("")
              setExpandedZoneId(null)
            }}
            aria-label="Limpiar búsqueda"
            style={{
              position: "absolute",
              right: "10px",
              top: "50%",
              transform: "translateY(-50%)",
              border: "none",
              background: "transparent",
              fontSize: "16px",
              cursor: "pointer",
              color: "#6b7280",
              lineHeight: 1,
              padding: "4px",
            }}
          >
            ✕
          </button>
        )}
      </div>

      {alerts.length === 0 ? (
        <p style={{ color: "#6b7280" }}>
          {searchTerm.trim() ? "No se han encontrado zonas" : "No hay alertas activas"}
        </p>
      ) : (
        alerts.map((z) => {
          const zoneId = Number(z.zone_id)
          const zoneName = zoneNames[zoneId] || `Zona ${zoneId}`
          const score = Number(z.score)
          const label = getStressLabel(score)
          const labelColor = getStressColor(score)
          const isExpanded = expandedZoneId === zoneId

          return (
            <div
              key={zoneId}
              style={{
                marginBottom: "18px",
                paddingBottom: "16px",
                borderBottom: "1px solid #f0f0f0",
              }}
            >
              <strong
                style={{
                  color: primaryColor,
                  fontSize: "15px",
                  display: "block",
                  marginBottom: "4px",
                }}
              >
                {zoneName}
              </strong>

              <div style={{ color: labelColor, marginBottom: "4px" }}>{label}</div>

              <small style={{ display: "block", marginBottom: "10px" }}>
                Score: {score.toFixed(2)}
              </small>

              <button
                onClick={() => toggleDetails(zoneId)}
                style={{
                  border: "1px solid #d1d5db",
                  background: "white",
                  color: primaryColor,
                  borderRadius: "8px",
                  padding: "6px 10px",
                  fontSize: "12px",
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {isExpanded ? "Ocultar previsión" : "Ver previsión"}
              </button>

              {isExpanded && (
                <ForecastDetails
                  zoneId={zoneId}
                  zoneName={zoneName}
                  score={score}
                  primaryColor={primaryColor}
                  dayOfWeek={dayOfWeek}
                  hour={hour}
                />
              )}
            </div>
          )
        })
      )}
    </div>
  )
}