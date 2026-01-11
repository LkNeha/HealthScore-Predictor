import React from "react";
import AnalyticsDashboard from "./pages/AnalyticsDashboard";
// import { useEffect, useState } from "react";



function App() {
  //   const [status, setStatus] = useState("Checking backend...");

  // useEffect(() => {
  //   async function check() {
  //     try {
  //       const data = await pingBackend();
  //       setStatus(`Backend: ${data.status} - ${data.message}`);
  //     } catch (err) {
  //       console.error(err);
  //       setStatus("Backend error / not reachable");
  //     }
  //   }
  //   check();
  // }, []);
return <AnalyticsDashboard />;
}

export default App;