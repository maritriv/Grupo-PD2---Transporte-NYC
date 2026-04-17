import { useEffect, useMemo, useState } from "react"

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

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value))
}

function formatHourLabel(date) {
  return new Intl.DateTimeFormat("es-ES", {
    hour: "2-digit",
    minute: "2-digit",
  }).format(date)
}

function buildForecast(score) {
  const now = new Date()
  const offsets = [0, 2, 4, 6, 12, 24]

  return offsets.map((offset, index) => {
    const date = new Date(now)
    date.setHours(date.getHours() + offset)

    const wave = Math.sin((index / (offsets.length - 1)) * Math.PI) * 0.12
    const trend = index < 3 ? 0.04 : -0.03
    const simulated = clamp(score + wave + trend)

    return {
      hourOffset: offset,
      timeLabel: offset === 0 ? "Ahora" : `+${offset}h`,
      clockLabel: formatHourLabel(date),
      score: Number(simulated.toFixed(2)),
      label: getStressLabel(simulated),
      color: getStressColor(simulated),
    }
  })
}

function ForecastDetails({ zoneName, score, primaryColor }) {
  const forecast = useMemo(() => buildForecast(score), [score])

  const peak = forecast.reduce((best, item) =>
    item.score > best.score ? item : best
  )

  const average =
    forecast.reduce((acc, item) => acc + item.score, 0) / forecast.length

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
        <div
          style={{
            padding: "10px",
            borderRadius: "10px",
            background: "white",
            border: "1px solid #e5e7eb",
          }}
        >
          <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>
            Estado actual
          </div>
          <div style={{ color: getStressColor(score), fontWeight: 700 }}>
            {getStressLabel(score)}
          </div>
        </div>

        <div
          style={{
            padding: "10px",
            borderRadius: "10px",
            background: "white",
            border: "1px solid #e5e7eb",
          }}
        >
          <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>
            Score actual
          </div>
          <div style={{ color: primaryColor, fontWeight: 700 }}>
            {score.toFixed(2)}
          </div>
        </div>

        <div
          style={{
            padding: "10px",
            borderRadius: "10px",
            background: "white",
            border: "1px solid #e5e7eb",
          }}
        >
          <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>
            Pico previsto
          </div>
          <div style={{ color: getStressColor(peak.score), fontWeight: 700 }}>
            {peak.score.toFixed(2)}
          </div>
        </div>

        <div
          style={{
            padding: "10px",
            borderRadius: "10px",
            background: "white",
            border: "1px solid #e5e7eb",
          }}
        >
          <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>
            Hora punta estimada
          </div>
          <div style={{ color: primaryColor, fontWeight: 700 }}>
            {peak.timeLabel} · {peak.clockLabel}
          </div>
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
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "13px",
          }}
        >
          <thead>
            <tr style={{ background: "#f3f4f6", color: "#374151" }}>
              <th style={{ textAlign: "left", padding: "8px 10px" }}>Horizonte</th>
              <th style={{ textAlign: "left", padding: "8px 10px" }}>Hora</th>
              <th style={{ textAlign: "left", padding: "8px 10px" }}>Score</th>
              <th style={{ textAlign: "left", padding: "8px 10px" }}>Estado</th>
            </tr>
          </thead>
          <tbody>
            {forecast.map((item) => (
              <tr key={item.timeLabel} style={{ borderTop: "1px solid #e5e7eb" }}>
                <td style={{ padding: "8px 10px" }}>{item.timeLabel}</td>
                <td style={{ padding: "8px 10px" }}>{item.clockLabel}</td>
                <td style={{ padding: "8px 10px" }}>{item.score.toFixed(2)}</td>
                <td style={{ padding: "8px 10px", color: item.color, fontWeight: 600 }}>
                  {item.label}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div
        style={{
          marginTop: "10px",
          fontSize: "12px",
          color: "#6b7280",
        }}
      >
        Media prevista próximas horas: <strong>{average.toFixed(2)}</strong>
      </div>
    </div>
  )
}

export default function AlertsPanel({ zones, primaryColor }) {
  const [zoneNames, setZoneNames] = useState({})
  const [expandedZoneId, setExpandedZoneId] = useState(null)

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
    return [...zones]
      .filter((z) => Number(z.score) >= 0.6)
      .sort((a, b) => Number(b.score) - Number(a.score))
  }, [zones])

  function toggleDetails(zoneId) {
    setExpandedZoneId((current) => (current === zoneId ? null : zoneId))
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
      <h3 style={{ marginTop: 0, color: primaryColor }}>
        Alertas
      </h3>

      {alerts.length === 0 ? (
        <p style={{ color: "#6b7280" }}>No hay alertas activas</p>
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

              <div style={{ color: labelColor, marginBottom: "4px" }}>
                {label}
              </div>

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
                  zoneName={zoneName}
                  score={score}
                  primaryColor={primaryColor}
                />
              )}
            </div>
          )
        })
      )}
    </div>
  )
}