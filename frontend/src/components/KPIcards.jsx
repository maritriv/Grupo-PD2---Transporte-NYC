export default function KPICards({ zones }) {
  const totalZones = zones.length
  const highZones = zones.filter((z) => z.level === "high").length
  const avgScore =
    zones.length > 0
      ? (zones.reduce((acc, z) => acc + z.score, 0) / zones.length).toFixed(2)
      : "0.00"

  const maxZone =
    zones.length > 0
      ? zones.reduce((max, z) => (z.score > max.score ? z : max), zones[0])
      : null

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: "14px",
      }}
    >
      <Card title="Zonas analizadas" value={totalZones} />
      <Card title="Zonas críticas" value={highZones} />
      <Card title="Score medio" value={avgScore} />
      <Card title="Zona más inestable" value={maxZone ? `Zona ${maxZone.zone_id}` : "-"} />
    </div>
  )
}

function Card({ title, value }) {
  return (
    <div
      style={{
        background: "linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95))",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: "14px",
        padding: "16px",
        color: "white",
      }}
    >
      <div style={{ fontSize: "14px", color: "#cbd5e1" }}>{title}</div>
      <div style={{ fontSize: "28px", fontWeight: "bold", marginTop: "8px" }}>
        {value}
      </div>
    </div>
  )
}