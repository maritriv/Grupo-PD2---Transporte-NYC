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
        gap: "12px",
        marginTop: "20px",
      }}
    >
      <Card title="Zonas analizadas" value={totalZones} />
      <Card title="Zonas críticas" value={highZones} />
      <Card title="Score medio" value={avgScore} />
      <Card
        title="Zona más inestable"
        value={maxZone ? `Zona ${maxZone.zone_id}` : "-"}
      />
    </div>
  )
}

function Card({ title, value }) {
  return (
    <div
      style={{
        background: "white",
        border: "1px solid #ddd",
        borderRadius: "12px",
        padding: "16px",
        boxShadow: "0 1px 4px rgba(0,0,0,0.05)",
      }}
    >
      <div style={{ fontSize: "14px", color: "#666" }}>{title}</div>
      <div style={{ fontSize: "24px", fontWeight: "bold", marginTop: "8px" }}>
        {value}
      </div>
    </div>
  )
}