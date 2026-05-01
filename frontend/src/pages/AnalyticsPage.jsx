import { useEffect, useMemo, useState } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
} from "recharts"
import { getMapData } from "../api/client"

const PRIMARY_COLOR = "#162a5a"

function getStressLabel(score) {
  if (score <= 0.2) return "Estable"
  if (score <= 0.4) return "Estrés bajo"
  if (score <= 0.6) return "Estrés moderado"
  if (score <= 0.8) return "Estrés alto"
  return "Crítico"
}

function getStressColor(label) {
  if (label === "Estable") return "#6cc36a"
  if (label === "Estrés bajo") return "#84a63a"
  if (label === "Estrés moderado") return "#c98b1f"
  if (label === "Estrés alto") return "#d97a45"
  return "#dc2626"
}

function getCurrentModelDate() {
  const now = new Date()
  const dayOfWeek = (now.getDay() + 6) % 7
  const hour = now.getHours()
  return { dayOfWeek, hour }
}

export default function AnalyticsPage() {
  const [zones, setZones] = useState([])
  const [zoneNames, setZoneNames] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true)
        setError("")

        const { dayOfWeek, hour } = getCurrentModelDate()
        const data = await getMapData(dayOfWeek, hour)

        setZones(data?.zones || [])

        const geoRes = await fetch("/nyc-zones.geojson")
        const geoData = await geoRes.json()

        const mapping = {}
        for (const feature of geoData.features || []) {
          const props = feature.properties || {}
          const id = Number(
            props.LocationID ??
              props.zone_id ??
              props.ZoneID ??
              props.OBJECTID ??
              props.objectid ??
              props.id
          )

          const name = props.zone || props.Zone || `Zona ${id}`

          if (!Number.isNaN(id)) {
            mapping[id] = name
          }
        }

        setZoneNames(mapping)
      } catch (err) {
        console.error(err)
        setError("No se pudo cargar el análisis")
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])

  const analytics = useMemo(() => {
    const levels = {
      Estable: 0,
      "Estrés bajo": 0,
      "Estrés moderado": 0,
      "Estrés alto": 0,
      Crítico: 0,
    }

    for (const zone of zones) {
      const label = getStressLabel(Number(zone.score))
      levels[label] += 1
    }

    const distribution = Object.entries(levels).map(([label, count]) => ({
      label,
      count,
      color: getStressColor(label),
    }))

    const topStress = [...zones]
      .sort((a, b) => Number(b.score) - Number(a.score))
      .slice(0, 5)
      .map((z) => ({
        name: zoneNames[Number(z.zone_id)] || `Zona ${z.zone_id}`,
        score: Number(z.score),
      }))

    const topStable = [...zones]
      .sort((a, b) => Number(a.score) - Number(b.score))
      .slice(0, 5)
      .map((z) => ({
        name: zoneNames[Number(z.zone_id)] || `Zona ${z.zone_id}`,
        score: Number(z.score),
      }))

    const avgScore =
      zones.length > 0
        ? zones.reduce((acc, z) => acc + Number(z.score), 0) / zones.length
        : 0

    const criticalZones = zones.filter((z) => Number(z.score) > 0.8).length
    const stableZones = zones.filter((z) => Number(z.score) <= 0.2).length

    return {
      distribution,
      topStress,
      topStable,
      avgScore,
      criticalZones,
      stableZones,
    }
  }, [zones, zoneNames])

  if (loading) {
    return (
      <PageLayout>
        <p>Cargando análisis...</p>
      </PageLayout>
    )
  }

  if (error) {
    return (
      <PageLayout>
        <p style={{ color: "#dc2626" }}>{error}</p>
      </PageLayout>
    )
  }

  return (
    <PageLayout>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "24px",
        }}
      >
        <div>
          <h1 style={{ margin: 0, color: PRIMARY_COLOR }}>Análisis completo</h1>
          <p style={{ margin: "6px 0 0", color: "#6b7280" }}>
            Resumen avanzado del estrés urbano por zonas
          </p>
        </div>

        <button
          onClick={() => (window.location.href = "/")}
          style={{
            border: "1px solid #d1d5db",
            background: "white",
            color: PRIMARY_COLOR,
            borderRadius: "10px",
            padding: "10px 14px",
            fontWeight: 700,
            cursor: "pointer",
          }}
        >
          Volver al dashboard
        </button>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "16px",
          marginBottom: "20px",
        }}
      >
        <SummaryCard title="Score medio" value={analytics.avgScore.toFixed(2)} />
        <SummaryCard title="Zonas críticas" value={analytics.criticalZones} />
        <SummaryCard title="Zonas estables" value={analytics.stableZones} />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "20px",
        }}
      >
        <ChartCard title="Distribución por nivel de estrés">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={analytics.distribution}>
              <CartesianGrid stroke="#eeeeee" />
              <XAxis dataKey="label" tick={{ fontSize: 12 }} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count">
                {analytics.distribution.map((entry) => (
                  <Cell key={entry.label} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Peso relativo de cada nivel">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={analytics.distribution}
                dataKey="count"
                nameKey="label"
                outerRadius={95}
                label
              >
                {analytics.distribution.map((entry) => (
                  <Cell key={entry.label} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Top 5 zonas más inestables">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={analytics.topStress} layout="vertical">
              <CartesianGrid stroke="#eeeeee" />
              <XAxis type="number" domain={[0, 1]} />
              <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="score" fill="#dc2626" />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Top 5 zonas más estables">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={analytics.topStable} layout="vertical">
              <CartesianGrid stroke="#eeeeee" />
              <XAxis type="number" domain={[0, 1]} />
              <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="score" fill="#22c55e" />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </PageLayout>
  )
}

function PageLayout({ children }) {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f6f7fb",
        padding: "28px",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <div style={{ maxWidth: "1400px", margin: "0 auto" }}>{children}</div>
    </div>
  )
}

function SummaryCard({ title, value }) {
  return (
    <div
      style={{
        background: "linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95))",
        color: "white",
        borderRadius: "16px",
        padding: "20px",
      }}
    >
      <div style={{ color: "#cbd5e1", fontSize: "14px" }}>{title}</div>
      <div style={{ fontSize: "34px", fontWeight: 800, marginTop: "8px" }}>{value}</div>
    </div>
  )
}

function ChartCard({ title, children }) {
  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "16px",
        padding: "20px",
      }}
    >
      <h3 style={{ marginTop: 0, color: PRIMARY_COLOR }}>{title}</h3>
      {children}
    </div>
  )
}