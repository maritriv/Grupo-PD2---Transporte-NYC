import { useState, useEffect } from "react"
import { getMapData } from "../api/client"
import Controls from "../components/Controls"
import MapView from "../components/MapView"

export default function Home() {

  const [day, setDay] = useState(2)
  const [hour, setHour] = useState(18)
  const [zones, setZones] = useState([])

  useEffect(() => {
    loadData()
  }, [day, hour])

  async function loadData() {
    const data = await getMapData(day, hour)
    setZones(data.zones)
  }

  return (
    <div>

      <h1>NYC Taxi Demand</h1>

      <Controls
        day={day}
        hour={hour}
        setDay={setDay}
        setHour={setHour}
      />

      <MapView zones={zones} />

    </div>
  )
}