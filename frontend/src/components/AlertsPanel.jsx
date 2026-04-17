import { useEffect, useMemo, useState } from "react"

export default function AlertsPanel({ zones, primaryColor }) {
  const [zoneNames, setZoneNames] = useState({})

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
      .filter((z) => z.level === "high")
      .sort((a, b) => Number(b.score) - Number(a.score))
  }, [zones])

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
          const zoneName = zoneNames[Number(z.zone_id)] || `Zona ${z.zone_id}`

          return (
            <div key={z.zone_id} style={{ marginBottom: "16px" }}>
              <strong
                style={{
                  color: primaryColor,
                  fontSize: "15px",
                }}
              >
                {zoneName}
              </strong>

              <div style={{ color: "#dc2626" }}>Estrés alto</div>

              <small>Score: {Number(z.score).toFixed(2)}</small>
            </div>
          )
        })
      )}
    </div>
  )
}