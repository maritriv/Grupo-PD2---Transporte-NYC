import { useEffect, useMemo, useState } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts"
import { getMapData } from "../api/client"

const PRIMARY_COLOR = "#162a5a"
const HOURS = [8, 10, 12, 14, 16, 18]

function getPressureLabel(score) {
  if (score <= 0.2) return "Muy recomendable"
  if (score <= 0.4) return "Buena opción"
  if (score <= 0.6) return "Normal"
  if (score <= 0.8) return "Puede haber espera"
  return "Mejor evitar ahora"
}

function getPressureColor(score) {
  if (score <= 0.2) return "#6cc36a"
  if (score <= 0.4) return "#84a63a"
  if (score <= 0.6) return "#c98b1f"
  if (score <= 0.8) return "#d97a45"
  return "#dc2626"
}

function formatPressure(score) {
  return `${Math.round(Number(score) * 100)}%`
}

function getCurrentModelDate() {
  const now = new Date()
  const dayOfWeek = (now.getDay() + 6) % 7
  const hour = now.getHours()
  return { dayOfWeek, hour }
}

export default function AnalyticsPage() {
  const [zones, setZones] = useState([])
  const [hourlyData, setHourlyData] = useState([])
  const [zoneNames, setZoneNames] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState("")

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true)
        setError("")

        const { dayOfWeek, hour } = getCurrentModelDate()
        const currentData = await getMapData(dayOfWeek, hour)
        setZones(currentData?.zones || [])

        const hourlyResults = await Promise.all(
          HOURS.map(async (h) => {
            const response = await getMapData(dayOfWeek, h)
            const hourZones = response?.zones || []

            const avg =
              hourZones.length > 0
                ? hourZones.reduce((acc, z) => acc + Number(z.score), 0) /
                  hourZones.length
                : 0

            return {
              hour: `${String(h).padStart(2, "0")}:00`,
              score: Number(avg.toFixed(2)),
            }
          })
        )

        setHourlyData(hourlyResults)

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
    const namedZones = zones.map((z) => ({
      id: Number(z.zone_id),
      name: zoneNames[Number(z.zone_id)] || `Zona ${z.zone_id}`,
      score: Number(z.score),
      rawStress: Number(z.raw_stress),
    }))

    const recommendedZones = [...namedZones]
      .sort((a, b) => a.score - b.score)
      .slice(0, 5)

    const avoidZones = [...namedZones]
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)

    const avgScore =
      namedZones.length > 0
        ? namedZones.reduce((acc, z) => acc + z.score, 0) / namedZones.length
        : 0

    const bestHour = [...hourlyData].sort((a, b) => a.score - b.score)[0]
    const worstHour = [...hourlyData].sort((a, b) => b.score - a.score)[0]

    return {
      recommendedZones,
      avoidZones,
      avgScore,
      bestHour,
      worstHour,
    }
  }, [zones, zoneNames, hourlyData])

  if (loading) {
    return (
      <PageLayout>
        <p>Cargando recomendaciones...</p>
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
      <Header />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "16px",
          marginBottom: "22px",
        }}
      >
        <SummaryCard
          title="Presión media ahora"
          value={formatPressure(analytics.avgScore)}
          subtitle={getPressureLabel(analytics.avgScore)}
        />

        <SummaryCard
          title="Mejor hora estimada"
          value={analytics.bestHour?.hour || "-"}
          subtitle="Menor presión media"
        />

        <SummaryCard
          title="Hora más saturada"
          value={analytics.worstHour?.hour || "-"}
          subtitle="Mayor presión media"
        />
      </div>

      <InfoCard
        title="¿Para qué sirve este análisis?"
        text="Esta página te ayuda a elegir mejor dónde y cuándo pedir un viaje. Una presión alta puede indicar más demanda, más tráfico, mayor espera o más probabilidad de precios elevados."
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "20px",
          marginTop: "20px",
        }}
      >
        <RecommendationCard
          title="Zonas recomendadas para pedir un viaje"
          subtitle="Áreas con menor presión en este momento."
          zones={analytics.recommendedZones}
          mode="good"
        />

        <RecommendationCard
          title="Zonas que conviene evitar"
          subtitle="Áreas con más saturación y posible subida de espera o precio."
          zones={analytics.avoidZones}
          mode="bad"
        />
      </div>

      <ChartCard
        title="Mejores momentos para moverse"
        subtitle="Compara la presión media estimada a distintas horas del día."
      >
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={hourlyData} margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
            <CartesianGrid stroke="#eeeeee" />
            <XAxis dataKey="hour" />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(value) => `${Math.round(value * 100)}%`}
            />
            <Tooltip
              formatter={(value) => [
                formatPressure(value),
                "Presión estimada",
              ]}
            />
            <Bar dataKey="score" radius={[8, 8, 0, 0]}>
              {hourlyData.map((entry) => (
                <Cell key={entry.hour} fill={getPressureColor(entry.score)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </PageLayout>
  )
}

function Header() {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "24px",
      }}
    >
      <div>
        <h1 style={{ margin: 0, color: PRIMARY_COLOR }}>Análisis para moverte mejor</h1>
        <p style={{ margin: "6px 0 0", color: "#6b7280" }}>
          Recomendaciones para elegir dónde y cuándo pedir un viaje en Nueva York
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

function SummaryCard({ title, value, subtitle }) {
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
      <div style={{ color: "#cbd5e1", fontSize: "13px", marginTop: "4px" }}>{subtitle}</div>
    </div>
  )
}

function InfoCard({ title, text }) {
  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "16px",
        padding: "18px 20px",
      }}
    >
      <h3 style={{ margin: "0 0 8px", color: PRIMARY_COLOR }}>{title}</h3>
      <p style={{ margin: 0, color: "#475569", lineHeight: 1.5 }}>{text}</p>
    </div>
  )
}

function RecommendationCard({ title, subtitle, zones, mode }) {
  const isGood = mode === "good"

  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "16px",
        padding: "20px",
      }}
    >
      <h3 style={{ margin: "0 0 6px", color: PRIMARY_COLOR }}>{title}</h3>
      <p style={{ margin: "0 0 16px", color: "#6b7280", fontSize: "14px" }}>{subtitle}</p>

      <div style={{ display: "grid", gap: "10px" }}>
        {zones.map((zone, index) => (
          <div
            key={zone.id}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              border: "1px solid #e5e7eb",
              borderRadius: "12px",
              padding: "12px",
              background: isGood ? "#f8fff8" : "#fff7f7",
            }}
          >
            <div>
              <div style={{ fontWeight: 800, color: PRIMARY_COLOR }}>
                {index + 1}. {zone.name}
              </div>
              <div style={{ color: "#6b7280", fontSize: "13px", marginTop: "3px" }}>
                {getPressureLabel(zone.score)}
              </div>
            </div>

            <div style={{ textAlign: "right" }}>
              <div
                style={{
                  fontWeight: 800,
                  color: getPressureColor(zone.score),
                  fontSize: "18px",
                }}
              >
                Índice {formatPressure(zone.score)}
              </div>

              <div
                style={{
                  color: "#6b7280",
                  fontSize: "12px",
                  marginTop: "3px",
                }}
              >
                Stress real{" "}
                {Number.isFinite(zone.rawStress)
                  ? zone.rawStress.toFixed(2)
                  : "-"}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function ChartCard({ title, subtitle, children }) {
  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "16px",
        padding: "20px",
        marginTop: "20px",
      }}
    >
      <h3 style={{ margin: "0 0 6px", color: PRIMARY_COLOR }}>{title}</h3>
      <p style={{ margin: "0 0 16px", color: "#6b7280", fontSize: "14px" }}>
        {subtitle}
      </p>
      {children}
    </div>
  )
}