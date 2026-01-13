import React, { useState, useEffect, useRef } from "react";
import "../index.css";
import { runPrediction } from "../services/apiClient";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
  ComposedChart,
} from "recharts";
import CountUp from "react-countup";
export default function InspectorDashboard({ onBack }) {
  const [step, setStep] = useState("choice"); // choice, auth, risk, predict
  const [selectedAction, setSelectedAction] = useState(null);
  const [code, setCode] = useState("");
  const [showAnswer, setShowAnswer] = useState(false);

  const handleAuth = () => {
    if (code.length === 6) {
      setStep(selectedAction);
    } else {
      alert("Please enter a 6-digit number");
    }
  };
  // const resetFlow = () => {
  //   setStep("choice");
  //   setSelectedAction(null);
  //   setCode("");
  //   setShowAnswer(false);
  // };
  const [form, setForm] = useState({
    businessName: "",
    inspectionDate: "",
    inspectionType: "",
  });
  

  const [insights, setInsights] = useState(null);
  const [insightsError, setInsightsError] = useState(null);
  const riskRef = useRef(null);
  const predictRef = useRef(null);

  useEffect(() => {
    if (step !== "risk") return;
    const load = async () => {
      try {
        const res = await fetch("http://localhost:8000/inspector-insights");
        const data = await res.json();
        console.log("/inspector-insights payload", data);
        setInsights(data);
      } catch (e) {
        setInsightsError(e.message);
      }
    };
    load();
  }, [step]);

  // Auto-scroll to the relevant section when step changes
  useEffect(() => {
    if (step === "risk" && riskRef.current) {
      riskRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    } else if (step === "predict" && predictRef.current) {
      predictRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [step]);
  const trends = insights?.inspection_trends || [];
  
  const delayStats = insights?.inspection_delay_stats || [];
  
  const riskDist = insights?.inspection_year_pie || [];
  
  const hasYearPie = Array.isArray(riskDist) && riskDist.length > 0;
  
  const highRisk = insights?.high_risk_records || [];
  const [highRiskLimit, setHighRiskLimit] = useState(10);
  const highRiskToShow = highRisk.slice(0, highRiskLimit);
  const summary = insights?.summary || {};
  const totalInspections = summary.total_inspections;
  const overallFailRate = summary.overall_fail_rate;
  const highRiskRestaurants = summary.high_risk_restaurants;
  const lowRiskRestaurants = summary.low_risk_restaurants;
  const pieColors = [
    "#2563eb", // blue
    "#22c55e", // green
    "#f97316", // orange
    "#e11d48", // red
    "#a855f7", // purple
    "#14b8a6", // teal
  ];

  const [prediction, setPrediction] = useState(null);   // {risk_label, risk_score}
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const updateField = (field) => (e) => {
    setForm((f) => ({ ...f, [field]: e.target.value }));
  };
  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setShowAnswer(false);

    // Derive date-based features from the selected inspection date
    const inspectionDateObj = form.inspectionDate
      ? new Date(form.inspectionDate)
      : null;

    const isValidDate = (d) => d instanceof Date && !Number.isNaN(d.getTime());

    let inspYear = 0;
    let inspMonth = 0;
    let inspDay = 0;
    let inspDow = 0; // 0 (Sunday) - 6 (Saturday)

    if (isValidDate(inspectionDateObj)) {
      inspYear = inspectionDateObj.getFullYear();
      inspMonth = inspectionDateObj.getMonth() + 1; // JS months are 0-based
      inspDay = inspectionDateObj.getDate();
      inspDow = inspectionDateObj.getDay();
    }

    try {
      const data = await runPrediction({
        business_name: form.businessName,
        inspection_type: form.inspectionType,
        inspection_date: form.inspectionDate,
      });

      setPrediction(data);
      setShowAnswer(true);
    } catch (e) {
      console.error(e);
      setError("Unable to run prediction.");
    } finally {
      setLoading(false);
    }
  };



  return (
    <div className="dashboard">


      {step === "choice" && (
        <div className="modal-overlay">
          <div className="modal-box">
            <h2>Inspector Options</h2>
            <button
              onClick={() => {
                setSelectedAction("risk");
                setStep("auth");
              }}
            >
              Restaurant at Risk
            </button>

            <button
              className="secondary"
              onClick={() => {
                setSelectedAction("predict");
                setStep("auth");
              }}
            >
              Predict
            </button>
            <button
              className="secondary"
              onClick={onBack}
            >
              Back
            </button>
          </div>
        </div>
      )}


      {step === "auth" && (
        <div className="modal-overlay">
          <div className="modal-box">
            <h2>Enter your Inspection id</h2>
            <input
              type="number"
              placeholder="000000"
              value={code}
              onChange={(e) => setCode(e.target.value)}
            />
            <button onClick={handleAuth}>Continue</button>
          </div>
        </div>
      )}


      {step === "risk" && (
        <div ref={riskRef} className="prediction-panel" style={{ marginTop: 0 }}>
          <div className="prediction-header">
            <div>
              <h2 className="title">INSPECTOR ANALYSIS</h2>
              <p className="prediction-sub">
                Overview of inspection trends, types, and high‑risk businesses.
              </p>
            </div>
          </div>
          {insightsError && <p className="error-text">{insightsError}</p>}
          {insights && (
            <div
              className="prediction-grid"
              style={{
                marginTop: 16,
                marginBottom: 8,
                gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
              }}
            >
              <div className="panel" style={{ padding: "12px 16px" }}>
                <p className="prediction-sub" style={{ marginBottom: 4 }}>
                  Total inspections
                </p>
                <h3 style={{ fontSize: "2.4rem", fontWeight: 600 }}>
                  {totalInspections != null ? (
                    <CountUp end={totalInspections} duration={2} separator="," />
                  ) : (
                    "-"
                  )}
                </h3>
              </div>
              <div className="panel" style={{ padding: "12px 16px" }}>
                <p className="prediction-sub" style={{ marginBottom: 4 }}>
                  Overall fail rate
                </p>
                <h3 style={{ fontSize: "2.4rem", fontWeight: 600 }}>
                  {overallFailRate != null ? (
                    <CountUp
                      end={overallFailRate * 100}
                      decimals={1}
                      suffix="%"
                      duration={2}
                    />
                  ) : (
                    "-"
                  )}
                </h3>
              </div>
              <div className="panel" style={{ padding: "12px 16px" }}>
                <p className="prediction-sub" style={{ marginBottom: 4 }}>
                  High-risk restaurants
                </p>
                <h3 style={{ fontSize: "2.4rem", fontWeight: 600 }}>
                  {highRiskRestaurants != null ? (
                    <CountUp end={highRiskRestaurants} duration={2} separator="," />
                  ) : (
                    "-"
                  )}
                </h3>
              </div>
              <div className="panel" style={{ padding: "12px 16px" }}>
                <p className="prediction-sub" style={{ marginBottom: 4 }}>
                  Low-risk restaurants
                </p>
                <h3 style={{ fontSize: "2.4rem", fontWeight: 600 }}>
                  {lowRiskRestaurants != null ? (
                    <CountUp end={lowRiskRestaurants} duration={2} separator="," />
                  ) : (
                    "-"
                  )}
                </h3>
              </div>
            </div>
          )}

          <div className="dashboard-grid">
            {/* Inspector Trends (Line Chart) */}
            <div className="panel">
              <h3>Inspection Trends</h3>
              <ResponsiveContainer width="100%" height={370}>
                <LineChart data={trends}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="year"
                    interval={0}
                    tick={{ fontSize: 12, fill: "#111827" }}
                    stroke="#111827"
                  />
                  <YAxis stroke="#111827" />
                  <Tooltip />
                  
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="total"
                    name="Total inspections"
                    stroke="#3b82f6"   /* primary accent */
                  />
                  <Line
                    type="monotone"
                    dataKey="fails"
                    name="Failures"
                    stroke="#ef4444"   /* fail accent */
                  />
                  <Line
                    type="monotone"
                    dataKey="passes"
                    name="Non-failures"
                    stroke="#22c55e"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Inspector Records (Table) */}
            <div
              className="panel"
              style={{ backgroundColor: "#d1d5db", color: "#111827" }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 8,
                }}
              >
                <h3 style={{ margin: 0 }}>High-Risk Restaurants</h3>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span className="prediction-sub">Show</span>
                  <select
                    value={highRiskLimit}
                    onChange={(e) => setHighRiskLimit(Number(e.target.value))}
                    style={{
                      padding: "4px 8px",
                      borderRadius: 8,
                      border: "1px solid #9ca3af",
                      fontSize: "0.8rem",
                    }}
                  >
                    <option value={10}>10</option>
                    <option value={20}>20</option>
                    <option value={30}>30</option>
                  </select>
                </div>
              </div>
              <div
                className="table-wrapper"
                style={{
                  maxHeight: 420,
                  minHeight: 420,
                  overflowY: "auto",
                }}
              >
                <table
                  className="simple-table"
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    borderRadius: "0.5rem",
                    overflow: "hidden",
                    border: "1px solid #111827",
                    backgroundColor: "#020617",
                  }}
                >
                  <thead>
                    <tr
                      style={{
                        backgroundColor: "#0b1120",
                        borderBottom: "1px solid #111827",
                      }}
                    >
                      {[
                        "Business",
                        "Fails",
                        "Fail rate",
                        "Avg violations",
                        "Year",
                      ].map((h) => (
                        <th
                          key={h}
                          style={{
                            padding: "10px 12px",
                            textAlign: "left",
                            fontWeight: 600,
                            fontSize: "0.85rem",
                            color: "#e5e7eb",
                            borderRight: "1px solid #111827",
                            position: "sticky",
                            top: 0,
                            backgroundColor: "#0b1120",
                            zIndex: 1,
                          }}
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {highRiskToShow.map((r, idx) => (
                      <tr
                        key={`${r.BusinessName || r.year || idx}`}
                        style={{
                            backgroundColor:
                              idx % 2 === 0 ? "#ffffff" : "#f9fafb",
                        }}
                      >
                        <td
                          style={{
                            padding: "8px 12px",
                            borderTop: "1px solid #111827",
                            borderRight: "1px solid #111827",
                          }}
                        >
                          {r.BusinessName}
                        </td>
                        <td
                          style={{
                            padding: "8px 12px",
                            borderTop: "1px solid #111827",
                            borderRight: "1px solid #111827",
                          }}
                        >
                          {r.fails != null ? r.fails : "-"}
                        </td>
                        <td
                          style={{
                            padding: "8px 12px",
                            borderTop: "1px solid #111827",
                            borderRight: "1px solid #111827",
                          }}
                        >
                          {r.fail_rate != null
                            ? `${(r.fail_rate * 100).toFixed(1)}%`
                            : "-"}
                        </td>
                        <td
                          style={{
                            padding: "8px 12px",
                            borderTop: "1px solid #111827",
                            borderRight: "1px solid #111827",
                          }}
                        >
                          {r.avg_violations != null
                            ? r.avg_violations.toFixed(2)
                            : "-"}
                        </td>
                        <td
                          style={{
                            padding: "8px 12px",
                            borderTop: "1px solid #111827",
                          }}
                        >
                          {r.year || "-"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Operational / Scheduling View: Heatmap Calendar */}
            {/* <div className="panel">
              <h3>Risky Times: Heatmap</h3>
              <p className="prediction-sub" style={{ marginBottom: 8 }}>
                Axes = day of week vs month; darker cells mean higher fail rate.
              </p>
              {dowMonthHeatmap.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <ScatterChart
                    margin={{ top: 10, right: 10, bottom: 10, left: -10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      type="number"
                      dataKey="month_index"
                      name="Month"
                      domain={[1, 12]}
                      tickCount={12}
                      stroke="#111827"
                      tick={{ fontSize: 11, fill: "#111827" }}
                      tickFormatter={(v) => {
                        const labels = [
                          "Jan",
                          "Feb",
                          "Mar",
                          "Apr",
                          "May",
                          "Jun",
                          "Jul",
                          "Aug",
                          "Sep",
                          "Oct",
                          "Nov",
                          "Dec",
                        ];
                        return labels[v - 1] || v;
                      }}
                    />
                    <YAxis
                      type="number"
                      dataKey="dow_index"
                      name="Day of Week"
                      domain={[0, 6]}
                      tickCount={7}
                      stroke="#111827"
                      tick={{ fontSize: 11, fill: "#111827" }}
                      tickFormatter={(v) => {
                        const labels = [
                          "Sun",
                          "Mon",
                          "Tue",
                          "Wed",
                          "Thu",
                          "Fri",
                          "Sat",
                        ];
                        return labels[v] || v;
                      }}
                    />
                    <ZAxis
                      type="number"
                      dataKey="total"
                      range={[80, 220]}
                      name="Inspections"
                    />
                    <Tooltip
                      cursor={{ strokeDasharray: "3 3" }}
                      formatter={(value, name, props) => {
                        if (name === "total") return [value, "Inspections"];
                        if (name === "fail_rate")
                          return [
                            `${(value * 100).toFixed(1)}%`,
                            "Average fail rate",
                          ];
                        if (name === "avg_violations")
                          return [
                            value.toFixed(2),
                            "Avg violations",
                          ];
                        return [value, name];
                      }}
                      labelFormatter={(_, payload) => {
                        if (!payload || !payload[0]) return "";
                        const p = payload[0].payload || {};
                        return `${p.dow} in ${p.month}`;
                      }}
                      contentStyle={{ fontSize: "0.8rem" }}
                    />
                    <Scatter data={dowMonthHeatmap} name="Fail rate by time">
                      {dowMonthHeatmap.map((d, idx) => {
                        const r = d.fail_rate || 0;
                        let color = "#22c55e";
                        if (r >= 0.5) color = "#b91c1c";
                        else if (r >= 0.25) color = "#f97316";
                        return <Cell key={`hm-${idx}`} fill={color} />;
                      })}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              ) : (
                <p className="prediction-sub">
                  Calendar heatmap data not available for this dataset.
                </p>
              )}
            </div> */}

            {/* Distribution Pies (Year + Violation Severity) */}
            <div className="panel">
              <h3>Risk Distributions</h3>
              <div>
                <h4
                  style={{
                    margin: "0 0 8px 0",
                    fontSize: "0.85rem",
                    color: "#9ca3af",
                  }}
                >
                  By Year
                </h4>
                {hasYearPie ? (
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Tooltip formatter={(v, n) => [`${v}`, n]} />
                      <Legend />
                      <Pie
                        data={riskDist}
                        dataKey="value"
                        nameKey="label"
                        outerRadius={110}
                        paddingAngle={3}
                      >
                        {riskDist.map((entry, index) => (
                          <Cell
                            key={`year-cell-${index}`}
                            fill={pieColors[index % pieColors.length]}
                          />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="prediction-sub">
                    No year distribution data available.
                  </p>
                )}
              </div>
            </div>

            {/* Operational / Scheduling View: Risk vs Days Since Last Inspection */}
            <div className="panel">
              <h3>Risk vs Time Since Last Inspection</h3>
              <p className="prediction-sub" style={{ marginBottom: 8 }}>
                Bars show inspection volume; line shows fail probability by delay bucket.
              </p>
              {delayStats.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <ComposedChart data={delayStats} margin={{ left: -10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      dataKey="bucket"
                      stroke="#111827"
                      tick={{ fontSize: 11, fill: "#111827" }}
                    />
                    <YAxis
                      yAxisId="left"
                      stroke="#111827"
                      tick={{ fontSize: 11, fill: "#111827" }}
                      allowDecimals={false}
                      name="Inspections"
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      stroke="#4b5563"
                      tick={{ fontSize: 11, fill: "#4b5563" }}
                      domain={[0, 1]}
                      tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                      name="Fail rate"
                    />
                    <Tooltip
                      formatter={(value, name) => {
                        if (name === "total") return [value, "Inspections"];
                        if (name === "fail_rate")
                          return [
                            `${(value * 100).toFixed(1)}%`,
                            "Fail rate",
                          ];
                        if (name === "fails") return [value, "Fails"];
                        return [value, name];
                      }}
                    />
                    <Legend />
                    <Bar
                      yAxisId="left"
                      dataKey="total"
                      name="Inspections"
                      fill="#3b82f6"
                      radius={[4, 4, 0, 0]}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="fail_rate"
                      name="Fail rate"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <p className="prediction-sub">
                  Delay bucket statistics not available for this dataset.
                </p>
              )}
            </div>
          </div>
        </div>
      )}


      {step === "predict" && (
        <div ref={predictRef} className="prediction-panel">
          <div className="prediction-header">
            <div>
              <h2>Pass or Closure?</h2>
              <p className="prediction-sub">
                Quickly estimate restaurant risk based on business and
                inspection details.
              </p>
            </div>
            {/* <button className="ghost-btn" onClick={resetFlow}>
      Back
    </button> */}
          </div>

          <div className="prediction-grid">
            <div className="field">
              <label>Business Name</label>
              <input
                placeholder="Enter business name"
                value={form.businessName}
                onChange={updateField("businessName")}
              />
            </div>

            <div className="field">
              <label>Date</label>
              <input
                type="date"
                placeholder="Select inspection date"
                value={form.inspectionDate}
                onChange={updateField("inspectionDate")}
              />
            </div>

            <div className="field">
              <label>Inspection Type</label>
              <select className="select-inspection"
                defaultValue="" value={form.inspectionType} onChange={updateField("inspectionType")}>
                <option value="" disabled >Select Inspection Type</option>
                <option>Change of Ownership</option>
                <option>Complaint Inspection</option>
                <option>Complaint Inspection (i)</option>
                <option>Complaint Inspection (r)</option>
                <option>Complaint Inspection reinspection/follow-up</option>
                <option>Foodborne Illness</option>
                <option>Foodborne Illness Investigation</option>
                <option>New Construction</option>
                <option>New Ownership</option>
                <option>New Ownership (i)</option>
                <option>New Ownership (r)</option>
                <option>New Ownership - followup</option>
                <option>Non-inspection site visit</option>
                <option>Plan check</option>
                <option>Plan check (i)</option>
                <option>Plan check (r)</option>
                <option>Reinspection</option>
                <option>Reinspection/followup</option>
                <option>Routine Inspection</option>
                <option>Site visit</option>
                <option>Structural</option>
                <option>Structural inspection</option>
                <option>Nan</option>
              </select>
            </div>
          </div>

          <button
            className="primary-btn"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? "Running..." : "Run prediction"}
          </button>

          {error && <p className="error-text">{error}</p>}

          {showAnswer && prediction && (
            <div className="answer">
              <h3>Prediction Result</h3>
              <p>
                Restaurant Risk:{" "}
                <span
                  className={
                    prediction.risk_label === "HIGH"
                      ? "risk-high"
                      : prediction.risk_label === "MEDIUM"
                        ? "risk-medium"
                        : "risk-low"
                  }
                >
                  {prediction.risk_label}
                </span>
              </p>
              {/* <p>Score: {(prediction.risk_score * 100).toFixed(1)}%</p> */}
              <p>
                Predicted outcome: {prediction.outcome_text} (flag {prediction.outcome_label})
              </p>
              <p>
                {/* Risk band: {prediction.risk_label} – {(prediction.risk_score * 100).toFixed(1)}% */}
              </p>
            </div>
          )}
        </div>

      )}
    </div>
  );
}
