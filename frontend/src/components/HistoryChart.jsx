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

export default function HistoryChart({ primaryColor }) {
  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e5e7eb",
        borderRadius: "16px",
        padding: "20px",
      }}
    >
      <h3 style={{ marginTop: 0, color: primaryColor }}>
        Histórico
      </h3>

      <div style={{ width: "100%", height: 200 }}>
        <ResponsiveContainer>
          <LineChart data={data}>
            <CartesianGrid stroke="#eee" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Line type="monotone" dataKey="stress" stroke={primaryColor} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}