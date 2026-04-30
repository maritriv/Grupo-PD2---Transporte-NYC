const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api"

export async function getMapData(dayOfWeek, hour) {
  const res = await fetch(
    `${API_BASE}/map?day_of_week=${dayOfWeek}&hour=${hour}`
  )

  if (!res.ok) {
    throw new Error("Error cargando mapa")
  }

  return await res.json()
}

export async function getZoneForecast(zoneId, dayOfWeek, hour) {
  const res = await fetch(
    `${API_BASE}/predict/forecast?zone_id=${zoneId}&day_of_week=${dayOfWeek}&hour=${hour}`
  )

  if (!res.ok) {
    throw new Error("Error cargando previsión")
  }

  return await res.json()
}