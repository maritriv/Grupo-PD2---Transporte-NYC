export default function Controls({day, hour, setDay, setHour}) {

  return (

    <div>

      <label>Day of week</label>

      <input
        type="number"
        min="0"
        max="6"
        value={day}
        onChange={(e) => setDay(e.target.value)}
      />

      <label>Hour</label>

      <input
        type="number"
        min="0"
        max="23"
        value={hour}
        onChange={(e) => setHour(e.target.value)}
      />

    </div>

  )
}