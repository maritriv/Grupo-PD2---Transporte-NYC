export default function AlertsPanel({ zones }) {
  const alerts = zones
    .filter((z) => z.level === "high")
    .sort((a, b) => b.score - a.score)

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
      <h3 style={{ marginTop: 0 }}>Alertas</h3>

      {alerts.length === 0 ? (
        <p>No hay alertas activas</p>
      ) : (
        alerts.map((alert) => (
          <div
            key={alert.zone_id}
            style={{
              padding: "12px 0",
              borderBottom: "1px solid #eee",
            }}
          >
            <div style={{ fontWeight: "bold" }}>Zona {alert.zone_id}</div>
            <div>Estrés alto detectado</div>
            <small>Score: {alert.score.toFixed(2)}</small>
          </div>
        ))
      )}
    </div>
  )
}