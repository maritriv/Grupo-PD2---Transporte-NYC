export default function AlertsPanel({ zones, primaryColor }) {
  const alerts = zones.filter((z) => z.level === "high")

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

      {alerts.map((z) => (
        <div key={z.zone_id} style={{ marginBottom: "12px" }}>
          <strong>Zona {z.zone_id}</strong>
          <div style={{ color: "#dc2626" }}>Estrés alto</div>
          <small>Score: {z.score.toFixed(2)}</small>
        </div>
      ))}
    </div>
  )
}