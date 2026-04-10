import axios from "axios"

const api = axios.create({
  baseURL: "http://127.0.0.1:8000/api",
})

export async function getMapData(day, hour) {
  const res = await api.get("/map", {
    params: {
      day_of_week: day,
      hour: hour,
    },
  })

  return res.data
}