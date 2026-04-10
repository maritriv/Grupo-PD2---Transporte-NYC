export default function Controls({ day, hour, setDay, setHour }) {
  return (
    <div style={{ display: "flex", gap: "16px", alignItems: "center", marginBottom: "20px" }}>
      <div>
        <label style={{ marginRight: "8px" }}>Day of week</label>
        <input
          type="number"
          min="0"
          max="6"
          value={day}
          onChange={(e) => setDay(Number(e.target.value))}
        />
      </div>

      <div>
        <label style={{ marginRight: "8px" }}>Hour</label>
        <input
          type="number"
          min="0"
          max="23"
          value={hour}
          onChange={(e) => setHour(Number(e.target.value))}
        />
      </div>
    </div>
  )
}