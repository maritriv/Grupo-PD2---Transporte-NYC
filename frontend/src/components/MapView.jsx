function getColor(level) {
  if (level === "high") return "#ef4444"
  if (level === "medium") return "#f59e0b"
  return "#22c55e"
}

export default function MapView({ zones, primaryColor }) {
  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "16px",
        padding: "20px",
      }}
    >
      <h2 style={{ marginTop: 0, color: primaryColor }}>
        Mapa de Estrés Urbano
      </h2>

      <p style={{ color: "#6b7280" }}>
        Visualización del nivel de tensión por zona
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "12px",
          marginTop: "16px",
        }}
      >
        {zones.map((z) => (
          <div
            key={z.zone_id}
            style={{
              background: getColor(z.level),
              color: "white",
              padding: "16px",
              borderRadius: "12px",
            }}
          >
            <strong>Zona {z.zone_id}</strong>
            <div>Score: {z.score.toFixed(2)}</div>
            <div>Nivel: {z.level}</div>
          </div>
        ))}
      </div>
    </div>
  )
}