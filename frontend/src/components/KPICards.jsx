import { useEffect, useMemo, useState } from "react"

function getStressLabel(score) {
  if (score === null || score === undefined || Number.isNaN(score)) return "Sin datos"
  if (score <= 0.2) return "Estable"
  if (score <= 0.4) return "Estrés bajo"
  if (score <= 0.6) return "Estrés moderado"
  if (score <= 0.8) return "Estrés alto"
  return "Crítico"
}

function getStressColor(score) {
  if (score === null || score === undefined || Number.isNaN(score)) return "#cbd5e1"
  if (score <= 0.2) return "#6cc36a"
  if (score <= 0.4) return "#a7c957"
  if (score <= 0.6) return "#e0a63a"
  if (score <= 0.8) return "#d97a45"
  return "#dc2626"
}

export default function KPICards({ zones }) {
  const [zoneNames, setZoneNames] = useState({})

  useEffect(() => {
    async function loadZoneNames() {
      try {
        const res = await fetch("/nyc-zones.geojson")
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

  const stats = useMemo(() => {
    const totalZones = zones.length

    const criticalZones = zones.filter((z) => Number(z.score) > 0.8).length

    const avgScoreNumber =
      zones.length > 0
        ? zones.reduce((acc, z) => acc + Number(z.score), 0) / zones.length
        : 0

    const maxZone =
      zones.length > 0
        ? zones.reduce(
            (max, z) => (Number(z.score) > Number(max.score) ? z : max),
            zones[0]
          )
        : null

    const maxZoneId = maxZone ? Number(maxZone.zone_id) : null
    const maxZoneScore = maxZone ? Number(maxZone.score) : null
    const maxZoneName = maxZoneId ? zoneNames[maxZoneId] || `Zona ${maxZoneId}` : "-"

    return {
      totalZones,
      criticalZones,
      avgScore: avgScoreNumber.toFixed(2),
      avgScoreLabel: getStressLabel(avgScoreNumber),
      avgScoreColor: getStressColor(avgScoreNumber),
      maxZoneName,
      maxZoneScore: maxZoneScore !== null ? maxZoneScore.toFixed(2) : "-",
    }
  }, [zones, zoneNames])

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: "14px",
      }}
    >
      <Card title="Zonas analizadas" value={stats.totalZones} />
      <Card title="Zonas críticas" value={stats.criticalZones} />
      <Card
        title="Score medio"
        value={stats.avgScore}
        subtitle={stats.avgScoreLabel}
        subtitleColor={stats.avgScoreColor}
      />
      <Card
        title="Zona más inestable"
        value={stats.maxZoneName}
        subtitle={`Score: ${stats.maxZoneScore}`}
        small
      />
    </div>
  )
}

function Card({ title, value, subtitle, subtitleColor = "#cbd5e1", small = false }) {
  return (
    <div
      style={{
        background: "linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95))",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: "14px",
        padding: "16px",
        color: "white",
        minHeight: "112px",
      }}
    >
      <div style={{ fontSize: "14px", color: "#cbd5e1" }}>{title}</div>

      <div
        style={{
          fontSize: small ? "22px" : "28px",
          fontWeight: "bold",
          marginTop: "8px",
          lineHeight: 1.15,
        }}
      >
        {value}
      </div>

      {subtitle && (
        <div
          style={{
            marginTop: "8px",
            fontSize: "13px",
            fontWeight: 600,
            color: subtitleColor,
          }}
        >
          {subtitle}
        </div>
      )}
    </div>
  )
}