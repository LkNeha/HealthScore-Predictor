import React, { useState, useEffect, useRef } from "react";
import "../index.css";
import { runPrediction, getInspectorInsights } from "../services/apiClient";
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

const FACILITY_STATUS_LABELS = {
  0: "Pass",
  1: "Conditional Pass",
  2: "Fail",
};

function renderInspectorStatus(statusLabel, numericValue) {
  if (statusLabel) {
    return statusLabel;
  }
  if (numericValue === null || numericValue === undefined) {
    return "-";
  }
  const parsed = Number(numericValue);
  if (Number.isNaN(parsed)) {
    return "-";
  }
  return FACILITY_STATUS_LABELS[parsed] || "Unknown";
}

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
    analysisNeighborhood: "",
    violationCount: "",
  });


  const [insights, setInsights] = useState(null);
  const [insightsError, setInsightsError] = useState(null);
  const riskRef = useRef(null);
  const predictRef = useRef(null);
  useEffect(() => {
    if (step !== "risk") return;
    const load = async () => {
      try {
        const data = await getInspectorInsights();
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
  const trendsWithTotal = trends.map((row) => ({
    ...row,
    total:
      (row?.pass_count || 0) +
      (row?.conditional_count || 0) +
      (row?.fail_count || 0),
  }));

  const delayStats = insights?.inspection_delay_stats || [];
  const neighborhoodFails = (insights?.neighborhood_fail_stats || []).slice(0, 12);

  const riskDist = insights?.inspection_year_pie || [];
  const failYearPie = trends
    .map((row) => ({
      label: row?.year,
      value: row?.fail_count || 0,
    }))
    .filter((row) => row.label != null && row.value > 0);
  const hasYearPie = failYearPie.length > 0;

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
        analysis_neighborhood: form.analysisNeighborhood,
        // Optional override: current inspection's violation count
        violation_count:
          form.violationCount !== "" && !Number.isNaN(Number(form.violationCount))
            ? Number(form.violationCount)
            : null,
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
  // Derive a display risk label: if the multiclass
  // prediction is Conditional Pass, show MEDIUM risk in UI
  const effectiveRiskLabel =
    prediction && prediction.predicted_class_label === "Conditional Pass"
      ? "MEDIUM"
      : prediction?.risk_label || null;
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
              Inspection history
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
                <LineChart data={trendsWithTotal} margin={{ top: 5, right: 30, left: 20, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis
                    dataKey="year"
                    interval={0}
                    minTickGap={10}
                    tick={{ fontSize: 12, fill: "#111827" }}
                    stroke="#111827"
                    padding={{ left: 10, right: 10 }}
                  />
                  <YAxis stroke="#111827" />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="pass_count"
                    name="Pass"
                    stroke="#22c55e"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="total"
                    name="Total inspections"
                    stroke="#3b82f6"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="conditional_count"
                    name="Conditional Pass"
                    stroke="#0077b6"
                    strokeWidth={2}
                  />
                  <Line
                    type="monotone"
                    dataKey="fail_count"
                    name="Fail"
                    stroke="#c1121f"
                    strokeWidth={2}
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
                        "Facility status",
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
                            borderRight: "1px solid #111827",
                          }}
                        >
                          {renderInspectorStatus(r.status_label, r.facility_rating_status)}
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
            {/* Distribution Pie: Fail Counts by Year */}
            <div className="panel">
              <h3>Risk Distributions</h3>
              <h3>Fail Distribution by Year</h3>
              <div>
                <h4
                  style={{
                    margin: "0 0 8px 0",
                    fontSize: "0.85rem",
                    color: "#9ca3af",
                  }}
                >
                  By Year
                  Inspections rated Fail (status 2)
                </h4>
                {hasYearPie ? (
                  <ResponsiveContainer width="100%" height={260}>
                    <PieChart>
                      <Tooltip formatter={(v, n) => [`${v}`, "Fails"]} />
                      <Legend />
                      <Pie
                        data={failYearPie}
                        dataKey="value"
                        nameKey="label"
                        outerRadius={110}
                        paddingAngle={3}
                      >
                        {failYearPie.map((entry, index) => (
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
                    No fail counts detected for the selected years.
                  </p>
                )}
              </div>
            </div>

            {/* Operational / Scheduling View: Risk vs Days Since Last Inspection */}
            <div className="panel">
              <h3>Neighborhood Fail Hotspots</h3>
              <p className="prediction-sub" style={{ marginBottom: 8 }}>
                Count of inspections rated as fail (status 2) by analysis neighborhood.
              </p>
              {neighborhoodFails.length > 0 ? (
                <div className="chart-scroll-wrapper">
                  <ResponsiveContainer
                    width={Math.min(1200, neighborhoodFails.length * 120)}
                    height={280}
                  >
                    <BarChart data={neighborhoodFails} margin={{ left: -10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis
                        dataKey="neighborhood"
                        stroke="#111827"
                        interval={0}
                        angle={-25}
                        textAnchor="end"
                        height={90}
                        tick={{ fontSize: 11, fill: "#111827" }}
                      />
                      <YAxis
                        yAxisId="left"
                        allowDecimals={false}
                        stroke="#111827"
                        tick={{ fontSize: 11, fill: "#111827" }}
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
                          if (name === "fails") return [value, "Fails (rating 2)"];
                          if (name === "total") return [value, "Total inspections"];
                          if (name === "fail_rate") {
                            return [`${(value * 100).toFixed(1)}%`, "Fail rate"];
                          }
                          return [value, name];
                        }}
                      />
                      <Legend />
                      <Bar
                        yAxisId="left"
                        dataKey="fails"
                        name="Fails (rating 2)"
                        fill="#ef4444"
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
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="prediction-sub">
                  Neighborhood fail distribution not available for this dataset.
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
              <label>Violation count (this inspection)</label>
              <input
                type="number"
                min="0"
                placeholder="e.g. 0, 1, 2"
                value={form.violationCount}
                onChange={updateField("violationCount")}
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

            <div className="field">
              <label>Neighbourhood (analysis)</label>
              <select
                value={form.analysisNeighborhood}
                onChange={updateField("analysisNeighborhood")}
              >
                <option value="">Select neighbourhood</option>
                <option value="analysis_neighborhood_Bernal Heights">Bernal Heights</option>
                <option value="analysis_neighborhood_Castro/Upper Market">Castro/Upper Market</option>
                <option value="analysis_neighborhood_Chinatown">Chinatown</option>
                <option value="analysis_neighborhood_Excelsior">Excelsior</option>
                <option value="analysis_neighborhood_Financial District/South Beach">Financial District/South Beach</option>
                <option value="analysis_neighborhood_Glen Park">Glen Park</option>
                <option value="analysis_neighborhood_Golden Gate Park">Golden Gate Park</option>
                <option value="analysis_neighborhood_Haight Ashbury">Haight Ashbury</option>
                <option value="analysis_neighborhood_Hayes Valley">Hayes Valley</option>
                <option value="analysis_neighborhood_Inner Richmond">Inner Richmond</option>
                <option value="analysis_neighborhood_Inner Sunset">Inner Sunset</option>
                <option value="analysis_neighborhood_Japantown">Japantown</option>
                <option value="analysis_neighborhood_Lakeshore">Lakeshore</option>
                <option value="analysis_neighborhood_Lincoln Park">Lincoln Park</option>
                <option value="analysis_neighborhood_Lone Mountain/USF">Lone Mountain/USF</option>
                <option value="analysis_neighborhood_Marina">Marina</option>
                <option value="analysis_neighborhood_McLaren Park">McLaren Park</option>
                <option value="analysis_neighborhood_Mission">Mission</option>
                <option value="analysis_neighborhood_Mission Bay">Mission Bay</option>
                <option value="analysis_neighborhood_Nob Hill">Nob Hill</option>
                <option value="analysis_neighborhood_Noe Valley">Noe Valley</option>
                <option value="analysis_neighborhood_North Beach">North Beach</option>
                <option value="analysis_neighborhood_Oceanview/Merced/Ingleside">Oceanview/Merced/Ingleside</option>
                <option value="analysis_neighborhood_Outer Mission">Outer Mission</option>
                <option value="analysis_neighborhood_Outer Richmond">Outer Richmond</option>
                <option value="analysis_neighborhood_Pacific Heights">Pacific Heights</option>
                <option value="analysis_neighborhood_Portola">Portola</option>
                <option value="analysis_neighborhood_Potrero Hill">Potrero Hill</option>
                <option value="analysis_neighborhood_Presidio">Presidio</option>
                <option value="analysis_neighborhood_Presidio Heights">Presidio Heights</option>
                <option value="analysis_neighborhood_Russian Hill">Russian Hill</option>
                <option value="analysis_neighborhood_Seacliff">Seacliff</option>
                <option value="analysis_neighborhood_South of Market">South of Market</option>
                <option value="analysis_neighborhood_Sunset/Parkside">Sunset/Parkside</option>
                <option value="analysis_neighborhood_Tenderloin">Tenderloin</option>
                <option value="analysis_neighborhood_Treasure Island">Treasure Island</option>
                <option value="analysis_neighborhood_Twin Peaks">Twin Peaks</option>
                <option value="analysis_neighborhood_Visitacion Valley">Visitacion Valley</option>
                <option value="analysis_neighborhood_West of Twin Peaks">West of Twin Peaks</option>
                <option value="analysis_neighborhood_Western Addition">Western Addition</option>
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
                    effectiveRiskLabel === "HIGH"
                      ? "risk-high"
                      : effectiveRiskLabel === "MEDIUM"
                        ? "risk-medium"
                        : "risk-low"
                  }
                >
                  {effectiveRiskLabel}
                </span>
              </p>
              {/* <p>
                Predicted outcome: {prediction.outcome_text} (flag {prediction.outcome_label})
              </p> */}
              {(prediction.predicted_class_label ||
                prediction.proba_pass != null ||
                prediction.proba_conditional != null ||
                prediction.proba_fail != null) && (
                <div style={{ marginTop: 12 }}>
                  {prediction.predicted_class_label && (
                    <p>
                      Multiclass prediction: {prediction.predicted_class_label}
                    </p>
                  )}
                  <p style={{ marginBottom: 4 }}>Class probabilities:</p>
                  <ul style={{ marginTop: 0, paddingLeft: 18 }}>
                    {prediction.proba_pass != null && (
                      <li>Pass: {(prediction.proba_pass * 100).toFixed(1)}%</li>
                    )}
                    {prediction.proba_conditional != null && (
                      <li>
                        Conditional Pass: {(prediction.proba_conditional * 100).toFixed(1)}%
                      </li>
                    )}
                    {prediction.proba_fail != null && (
                      <li>Fail: {(prediction.proba_fail * 100).toFixed(1)}%</li>
                    )}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}