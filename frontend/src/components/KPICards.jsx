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
  if (score <= 0.2) return "#22c55e" // verde
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

    const criticalZonesList = zones.filter((z) => Number(z.score) > 0.8)
    const stableZonesList = zones.filter((z) => Number(z.score) <= 0.2)

    const criticalZones = criticalZonesList.length
    const stableZones = stableZonesList.length

    const criticalPercentage =
      totalZones > 0 ? Math.round((criticalZones / totalZones) * 100) : 0

    const stablePercentage =
      totalZones > 0 ? Math.round((stableZones / totalZones) * 100) : 0

    const criticalAvg =
      criticalZones > 0
        ? criticalZonesList.reduce((acc, z) => acc + Number(z.score), 0) /
          criticalZones
        : 0

    const stableAvg =
      stableZones > 0
        ? stableZonesList.reduce((acc, z) => acc + Number(z.score), 0) /
          stableZones
        : 0

    const maxZone =
      zones.length > 0
        ? zones.reduce((max, z) =>
            Number(z.score) > Number(max.score) ? z : max
          )
        : null

    const minZone =
      zones.length > 0
        ? zones.reduce((min, z) =>
            Number(z.score) < Number(min.score) ? z : min
          )
        : null

    return {
      totalZones,

      criticalZones,
      criticalPercentage,
      criticalAvg: criticalAvg.toFixed(2),

      stableZones,
      stablePercentage,
      stableAvg: stableAvg.toFixed(2),

      maxZoneName:
        maxZone && zoneNames[maxZone.zone_id]
          ? zoneNames[maxZone.zone_id]
          : "-",
      maxZoneScore: maxZone ? Number(maxZone.score).toFixed(2) : "-",

      minZoneName:
        minZone && zoneNames[minZone.zone_id]
          ? zoneNames[minZone.zone_id]
          : "-",
      minZoneScore: minZone ? Number(minZone.score).toFixed(2) : "-",
    }
  }, [zones, zoneNames])

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
      {/* CRÍTICAS */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "14px" }}>
        <Card title="% zonas críticas" value={`${stats.criticalPercentage}%`} subtitle={`${stats.criticalZones} de ${stats.totalZones}`} />
        <Card title="Zonas críticas" value={stats.criticalZones} />
        <Card title="Score crítico medio" value={stats.criticalAvg} subtitle="Crítico" subtitleColor="#dc2626" />
        <Card title="Zona más inestable" value={stats.maxZoneName} subtitle={`Score: ${stats.maxZoneScore}`} small />
      </div>

      {/* ESTABLES */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "14px" }}>
        <Card title="% zonas estables" value={`${stats.stablePercentage}%`} subtitle={`${stats.stableZones} de ${stats.totalZones}`} />
        <Card title="Zonas estables" value={stats.stableZones} />
        <Card title="Score estable medio" value={stats.stableAvg} subtitle="Estable" subtitleColor="#22c55e" />
        <Card title="Zona más estable" value={stats.minZoneName} subtitle={`Score: ${stats.minZoneScore}`} small />
      </div>
    </div>
  )
}

function Card({
  title,
  value,
  subtitle,
  subtitleColor = "#cbd5e1",
  small = false,
}) {
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
      <div style={{ fontSize: "14px", color: "#cbd5e1" }}>
        {title}
      </div>

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