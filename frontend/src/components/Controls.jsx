export default function Controls({ day, hour, setDay, setHour }) {
  return (
    <div style={{ display: "flex", gap: "16px" }}>
      <div>
        <label style={{ fontSize: "12px", color: "#6b7280" }}>Día</label>
        <input
          type="number"
          min="0"
          max="6"
          value={day}
          onChange={(e) => setDay(Number(e.target.value))}
          style={{
            display: "block",
            padding: "8px",
            borderRadius: "8px",
            border: "1px solid #ccc",
          }}
        />
      </div>

      <div>
        <label style={{ fontSize: "12px", color: "#6b7280" }}>Hora</label>
        <input
          type="number"
          min="0"
          max="23"
          value={hour}
          onChange={(e) => setHour(Number(e.target.value))}
          style={{
            display: "block",
            padding: "8px",
            borderRadius: "8px",
            border: "1px solid #ccc",
          }}
        />
      </div>
    </div>
  )
}