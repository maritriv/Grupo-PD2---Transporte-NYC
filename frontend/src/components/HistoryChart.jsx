import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts"

const data = [
  { time: "08", stress: 0.2 },
  { time: "10", stress: 0.3 },
  { time: "12", stress: 0.4 },
  { time: "14", stress: 0.5 },
  { time: "16", stress: 0.7 },
  { time: "18", stress: 0.8 },
]

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
      <div>Score: {score.toFixed(2)}</div>
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

export default function HistoryChart({ primaryColor }) {
  const first = data[0]?.stress ?? 0
  const last = data[data.length - 1]?.stress ?? 0
  const trend = last > first ? "Tendencia creciente" : last < first ? "Tendencia descendente" : "Tendencia estable"
  const trendColor = getStressColor(last)

  function openAnalytics() {
    window.location.href = "/analytics"
  }

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

        <button
          onClick={openAnalytics}
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

      <div style={{ width: "100%", height: 220 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 10, right: 12, left: -18, bottom: 0 }}>
            <CartesianGrid stroke="#eeeeee" />
            <XAxis dataKey="time" />
            <YAxis domain={[0, 1]} />
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
    </div>
  )
}