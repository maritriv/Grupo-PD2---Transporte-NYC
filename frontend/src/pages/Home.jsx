import { useState, useEffect } from "react"
import { getMapData } from "../api/client"
import Controls from "../components/Controls"
import MapView from "../components/MapView"
import KPICards from "../components/KPICards"
import AlertsPanel from "../components/AlertsPanel"

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
    <div
      style={{
        padding: "24px",
        fontFamily: "Arial, sans-serif",
        background: "#f6f7fb",
        minHeight: "100vh",
      }}
    >
      <h1>NYC Taxi Demand</h1>

      <Controls
        day={day}
        hour={hour}
        setDay={setDay}
        setHour={setHour}
      />

      {loading && <p>Cargando datos...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {!loading && !error && (
        <>
          <KPICards zones={zones} />

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "2fr 1fr",
              gap: "20px",
              marginTop: "20px",
              alignItems: "start",
            }}
          >
            <MapView zones={zones} />
            <AlertsPanel zones={zones} />
          </div>
        </>
      )}
    </div>
  )
}