import { useState, useEffect } from "react"
import { getMapData } from "../api/client"
import Controls from "../components/Controls"
import MapView from "../components/MapView"
import KPICards from "../components/KPICards"
import AlertsPanel from "../components/AlertsPanel"
import HistoryChart from "../components/HistoryChart"

const PRIMARY_COLOR = "#162a5a"

export default function Home() {
  const now = new Date()

  const [day, setDay] = useState(now.getDay())
  const [hour, setHour] = useState(now.getHours())
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
      console.error(err)
      setError("Error cargando datos")
      setZones([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f6f7fb",
        color: "#111827",
        fontFamily: "Arial, sans-serif",
        padding: "24px",
      }}
    >
      <div style={{ maxWidth: "1400px", margin: "0 auto" }}>

        {/* HEADER */}
        <header
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "24px",
            padding: "16px 20px",
            borderRadius: "18px",
            background: "white",
            border: "1px solid #e5e7eb",
            boxShadow: "0 2px 8px rgba(0,0,0,0.05)",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "18px" }}>
            <img
              src="/macbrides.png"
              alt="MACBRIDES"
              style={{ height: "70px", objectFit: "contain" }}
            />

            <div>
              <h1
                style={{
                  margin: 0,
                  fontFamily: "'Cinzel', serif",
                  fontWeight: 700,
                  letterSpacing: "1px",
                  color: PRIMARY_COLOR,
                  fontSize: "32px",
                }}
              >
                MACBRIDES
              </h1>

              <p style={{ margin: 0, color: "#6b7280", fontSize: "14px" }}>
                NYC Urban Stress Dashboard
              </p>
            </div>
          </div>

          <Controls
            day={day}
            hour={hour}
            setDay={setDay}
            setHour={setHour}
          />
        </header>

        {loading && <p>Cargando...</p>}
        {error && <p style={{ color: "red" }}>{error}</p>}

        {!loading && !error && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "2fr 1fr",
              gap: "20px",
            }}
          >
            <div>
              <MapView zones={zones} primaryColor={PRIMARY_COLOR} />
              <div style={{ marginTop: "20px" }}>
                <KPICards zones={zones} primaryColor={PRIMARY_COLOR} />
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
              <AlertsPanel zones={zones} primaryColor={PRIMARY_COLOR} />
              <HistoryChart primaryColor={PRIMARY_COLOR} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}