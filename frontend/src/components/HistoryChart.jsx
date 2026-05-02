import { useEffect, useMemo, useState } from "react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Label,
} from "recharts"
import { getMapData } from "../api/client"

const HOURS = [8, 10, 12, 14, 16, 18]

function getStressLabel(score) {
  if (score <= 0.2) return "Estable"
  if (score <= 0.4) return "Estrés bajo"
  if (score <= 0.6) return "Estrés moderado"
  if (score <= 0.8) return "Estrés alto"
  return "Crítico"
}

function getStressColor(score) {
  if (score <= 0.2) return "#6cc36a"
  if (score <= 0.4) return "#84a63a"
  if (score <= 0.6) return "#c98b1f"
  if (score <= 0.8) return "#d97a45"
  return "#dc2626"
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null

  const score = Number(payload[0].value)

  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "10px",
        padding: "10px 12px",
        boxShadow: "0 6px 18px rgba(0,0,0,0.08)",
        fontSize: "13px",
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: "4px" }}>{label}:00</div>
      <div>Estrés medio: {score.toFixed(2)}</div>
      <div style={{ color: getStressColor(score), fontWeight: 700 }}>
        {getStressLabel(score)}
      </div>
    </div>
  )
}

function CustomDot({ cx, cy, payload }) {
  const color = getStressColor(Number(payload.stress))

  return (
    <circle
      cx={cx}
      cy={cy}
      r={5}
      fill="white"
      stroke={color}
      strokeWidth={3}
    />
  )
}

export default function HistoryChart({ primaryColor, dayOfWeek }) {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [showHelp, setShowHelp] = useState(false)

  useEffect(() => {
    async function loadHistory() {
      try {
        setLoading(true)
        setError("")

        const results = await Promise.all(
          HOURS.map(async (hour) => {
            const response = await getMapData(dayOfWeek, hour)
            const zones = response?.zones || []

            const avg =
              zones.length > 0
                ? zones.reduce((acc, z) => acc + Number(z.score), 0) / zones.length
                : 0

            return {
              time: String(hour).padStart(2, "0"),
              stress: Number(avg.toFixed(3)),
            }
          })
        )

        setData(results)
      } catch (err) {
        console.error(err)
        setError("No se pudo cargar el histórico")
        setData([])
      } finally {
        setLoading(false)
      }
    }

    loadHistory()
  }, [dayOfWeek])

  const trend = useMemo(() => {
    if (data.length < 2) return "Sin tendencia"

    const first = data[0].stress
    const last = data[data.length - 1].stress
    const diff = last - first

    if (Math.abs(diff) < 0.02) return "Nivel bastante estable"
    if (diff > 0) return "Sube durante el día"
    return "Baja durante el día"
  }, [data])

  const trendColor =
    data.length > 0 ? getStressColor(data[data.length - 1].stress) : "#6b7280"

  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "16px",
        padding: "20px",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: "12px",
          marginBottom: "12px",
        }}
      >
        <div>
          <h3 style={{ marginTop: 0, marginBottom: "6px", color: primaryColor }}>
            Histórico
          </h3>

          <div
            style={{
              fontSize: "13px",
              color: trendColor,
              fontWeight: 700,
            }}
          >
            {trend}
          </div>
        </div>

        <div
          style={{
            display: "flex",
            gap: "8px",
            flexWrap: "wrap",
            justifyContent: "flex-end",
          }}
        >
          <button
            onClick={() => setShowHelp((current) => !current)}
            style={{
              border: "1px solid #d1d5db",
              background: showHelp ? "#f3f4f6" : "white",
              color: primaryColor,
              borderRadius: "10px",
              padding: "8px 10px",
              fontSize: "12px",
              fontWeight: 700,
              cursor: "pointer",
              whiteSpace: "nowrap",
            }}
          >
            {showHelp ? "Ocultar ayuda" : "Ayuda"}
          </button>

          <button
            onClick={() => (window.location.href = "/analytics")}
            style={{
              border: "1px solid #d1d5db",
              background: "white",
              color: primaryColor,
              borderRadius: "10px",
              padding: "8px 10px",
              fontSize: "12px",
              fontWeight: 700,
              cursor: "pointer",
              whiteSpace: "nowrap",
            }}
          >
            Ver análisis completo
          </button>
        </div>
      </div>

      {showHelp && (
        <div
          style={{
            marginBottom: "14px",
            padding: "12px 14px",
            borderRadius: "12px",
            background: "#f8fafc",
            border: "1px solid #e5e7eb",
            color: "#334155",
            fontSize: "13px",
            lineHeight: 1.45,
          }}
        >
          <strong style={{ color: primaryColor }}>¿Qué significa?</strong>
          <br />
          Muestra cómo cambia el estrés medio de la ciudad a lo largo del día.
          Cuanto más cerca está de 1, mayor es la presión en las zonas analizadas.
        </div>
      )}

      {loading ? (
        <div style={{ color: "#6b7280", padding: "30px 0" }}>
          Cargando histórico...
        </div>
      ) : error ? (
        <div style={{ color: "#dc2626", padding: "30px 0" }}>{error}</div>
      ) : (
        <div style={{ width: "100%", height: 220 }}>
          <ResponsiveContainer>
            <LineChart
              data={data}
              margin={{ top: 8, right: 12, left: 8, bottom: 24 }}
            >
              <CartesianGrid stroke="#eeeeee" />

              <XAxis dataKey="time">
                <Label
                  value="Hora del día"
                  offset={-10}
                  position="insideBottom"
                  style={{ fontSize: 12, fill: "#6b7280" }}
                />
              </XAxis>

              <YAxis domain={[0, 1]}>
                <Label
                  value="Nivel de estrés"
                  angle={-90}
                  position="insideLeft"
                  style={{
                    textAnchor: "middle",
                    fontSize: 12,
                    fill: "#6b7280",
                  }}
                />
              </YAxis>

              <Tooltip content={<CustomTooltip />} />

              <Line
                type="monotone"
                dataKey="stress"
                stroke={primaryColor}
                strokeWidth={2}
                dot={<CustomDot />}
                activeDot={{ r: 7 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}