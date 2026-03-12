export default function MapView({zones}) {

  return (

    <div>

      <h2>Zones</h2>

      <ul>

        {zones.map(z => (
          <li key={z.zone_id}>
            Zone {z.zone_id} — score {z.score} — {z.level}
          </li>
        ))}

      </ul>

    </div>

  )
}