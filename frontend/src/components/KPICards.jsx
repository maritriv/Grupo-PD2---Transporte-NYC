import { useEffect, useMemo, useState } from "react"

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

    const avgScore =
      zones.length > 0
        ? (zones.reduce((acc, z) => acc + Number(z.score), 0) / zones.length).toFixed(2)
        : "0.00"

    const maxZone =
      zones.length > 0
        ? zones.reduce((max, z) =>
            Number(z.score) > Number(max.score) ? z : max
          , zones[0])
        : null

    const maxZoneId = maxZone ? Number(maxZone.zone_id) : null
    const maxZoneName = maxZoneId ? zoneNames[maxZoneId] || `Zona ${maxZoneId}` : "-"

    return {
      totalZones,
      criticalZones,
      avgScore,
      maxZoneName,
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
      <Card title="Score medio" value={stats.avgScore} />
      <Card title="Zona más inestable" value={stats.maxZoneName} small />
    </div>
  )
}

function Card({ title, value, small = false }) {
  return (
    <div
      style={{
        background: "linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95))",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: "14px",
        padding: "16px",
        color: "white",
        minHeight: "92px",
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
    </div>
  )
}