import Home from "./pages/Home"
import AnalyticsPage from "./pages/AnalyticsPage"

function App() {
  const path = window.location.pathname

  if (path === "/analytics") {
    return <AnalyticsPage />
  }

  return <Home />
}

export default App