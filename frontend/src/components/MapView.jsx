function getColor(level) {
  if (level === "high") return "#ef4444"     // rojo
  if (level === "medium") return "#f59e0b"   // naranja
  return "#22c55e"                           // verde
}

export default function MapView({ zones }) {
  return (
    <div
      style={{
        background: "white",
        border: "1px solid #ddd",
        borderRadius: "12px",
        padding: "16px",
        marginTop: "20px"
      }}
    >
      <h2 style={{ marginTop: 0 }}>Mapa de estrés urbano</h2>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "12px"
        }}
      >
        {zones.map((z) => (
          <div
            key={z.zone_id}
            style={{
              background: getColor(z.level),
              color: "white",
              padding: "14px",
              borderRadius: "10px"
            }}
          >
            <div style={{ fontWeight: "bold" }}>
              Zona {z.zone_id}
            </div>

            <div>Score: {z.score.toFixed(2)}</div>
            <div>Nivel: {z.level}</div>
          </div>
        ))}
      </div>
    </div>
  )
}