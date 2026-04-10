import { useState, useEffect } from "react"
import { getMapData } from "../api/client"
import Controls from "../components/Controls"
import MapView from "../components/MapView"

export default function Home() {
  const [day, setDay] = useState(2)
  const [hour, setHour] = useState(18)
  const [zones, setZones] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    loadData()
  }, [day, hour])

  async function loadData() {
    try {
      setLoading(true)
      setError("")

      const data = await getMapData(day, hour)
      console.log("MAP DATA:", data)

      setZones(data?.zones || [])
    } catch (err) {
      console.error("Error cargando mapa:", err)
      setError("No se pudieron cargar los datos del mapa")
      setZones([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: "24px", fontFamily: "Arial, sans-serif" }}>
      <h1>NYC Taxi Demand</h1>

      <Controls
        day={day}
        hour={hour}
        setDay={setDay}
        setHour={setHour}
      />

      {loading && <p>Cargando datos...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {!loading && !error && <MapView zones={zones} />}
    </div>
  )
}