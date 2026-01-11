// src/components/DataDashboard.jsx
import React, { useEffect, useState, useRef } from "react";
import CountUp from "react-countup";
import "../index.css"
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
} from "recharts";

export default function DataDashboard() {
  const [loading, setLoading] = useState(true);
  const [insights, setInsights] = useState(null);
  const [error, setError] = useState(null);
  const containerRef = useRef(null);

  useEffect(() => {
    const loadInsights = async () => {
      try {
        const res = await fetch("http://localhost:8000/data-insights");
        const data = await res.json();
        if (data.error) {
          setError(data.error);
        } else {
          setInsights(data);
        }
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };
    loadInsights();
  }, []);

  // Auto-scroll to the dashboard section once insights are loaded
  useEffect(() => {
    if (insights && containerRef.current) {
      containerRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [insights]);

  if (loading) return <div className="dashboard"><p>Loading model insights...</p></div>;
  if (error) return <div className="dashboard"><p>Error: {error}</p></div>;
  if (!insights) return null;

  const {
    roc_auc,
    recall_fail,
    total_records,
    n_features,
    roc_curve,
    feature_importance,
    correlations,
    fail_rate_by_year,
    classification_report,
  } = insights;

  // Build data for ROC chart
  const rocData =
    roc_curve?.fpr?.map((fprVal, idx) => ({
      fpr: fprVal,
      tpr: roc_curve.tpr[idx],
    })) || [];

  // Top 10 feature importance for bar chart
  const topFeatures = (feature_importance || []).slice(0, 10);

  // correlation heat data: flatten matrix into cells
  const corrFeatures = correlations.features;
  const corrValues = correlations.values;
  const corrData = [];
  corrFeatures.forEach((rowName, i) => {
    corrFeatures.forEach((colName, j) => {
      corrData.push({
        row: rowName,
        col: colName,
        value: corrValues[i][j],
      });
    });
  });


  // Class 1 metrics for bar chart
  const cls1 = classification_report?.["1"];
  const cls1Metrics = cls1
    ? [
      { metric: "precision", value: cls1.precision },
      { metric: "recall", value: cls1.recall },
      { metric: "f1-score", value: cls1["f1-score"] },
    ]
    : [];

  return (
    <section className="data-section" ref={containerRef}>
      <div className="dashboard data-dashboard">
        <div className="data-panel">
          <h2 className="title">MODEL INSIGHTS</h2>
          {/* Summary strip */}
          <div className="summary-row">
          <div className="summary-card">
            <div className="summary-label">Total Records</div>
            <div className="summary-value">
              <CountUp
                end={total_records || 0}
                duration={2}
                separator="," 
              />
            </div>
            <div className="summary-sub">Training / validation set</div>
          </div>

          <div className="summary-card">
            <div className="summary-label">Features</div>
            <div className="summary-value">
              <CountUp
                end={n_features || 0}
                duration={2}
              />
            </div>
            <div className="summary-sub">Engineered predictors</div>
          </div>

          <div className="summary-card">
            <div className="summary-label">ROC‑AUC</div>
            <div className="summary-value">
              <CountUp
                end={roc_auc || 0}
                duration={2}
                decimals={3}
              />
            </div>
            <div className="summary-sub">
              Recall (fail): {" "}
              <CountUp
                end={(recall_fail || 0) * 100}
                duration={2}
                decimals={1}
                suffix="%"
              />
            </div>
          </div>
          </div>

        {/* Block 1: AUC */}
        {/* <div className="none-data-card">
          
          <p className="auc-value">Overall AUC: {roc_auc.toFixed(3)}</p>
        </div> */}
        <div className="data-grid">
          {/* ROC line */}
          <div className="data-card">
            <h3>ROC Curve</h3>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={rocData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
                <XAxis
                  dataKey="fpr"
                  stroke="#111827"
                  tick={{ fill: "#111827" }}
                  tickFormatter={(v) => v.toFixed(3)}
                  label={{ value: "FPR", position: "insideBottomRight", dy: 10, fill: "#111827" }}
                />
                <YAxis
                  stroke="#111827"
                  tick={{ fill: "#111827" }}
                  label={{ value: "TPR", angle: -90, position: "insideLeft", fill: "#111827" }}
                />
                <Tooltip formatter={(v) => v.toFixed(3)} />
                <Legend verticalAlign="top" height={24} />
                <Line
                  type="monotone"
                  dataKey="tpr"
                  name="TPR"
                  stroke="#38bdf8"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>


          {/* Correlation heatmap grid */}
          <div className="data-card">
            <h3>Feature Correlations</h3>
            <div className="corr-heatmap">
              {/* header row */}
              <div className="corr-row corr-header">
                <div className="corr-cell corr-corner" />
                {corrFeatures.map((f) => (
                  <div key={`h-${f}`} className="corr-cell corr-header-cell">
                    {f}
                  </div>
                ))}
              </div>

              {/* matrix rows */}
              {corrValues.map((row, i) => (
                <div key={`r-${corrFeatures[i]}`} className="corr-row">
                  <div className="corr-cell corr-header-cell corr-row-label">
                    {corrFeatures[i]}
                  </div>
                  {row.map((val, j) => (
                    <div
                      key={`c-${i}-${j}`}
                      className="corr-cell corr-value-cell"
                      title={`${corrFeatures[i]} vs ${corrFeatures[j]}: ${val.toFixed(2)}`}
                      style={{
                        // Nude pastel tones for better readability
                        backgroundColor:
                          val >= 0
                            ? `rgba(250, 214, 195, ${Math.min(0.9, Math.abs(val) || 0.15)})` // warm nude for positive
                            : `rgba(214, 226, 255, ${Math.min(0.9, Math.abs(val) || 0.15)})`, // soft pastel for negative
                      }}
                    >
                      {val.toFixed(2)}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>


          {/* Feature importance */}
          <div className="data-card">
            <h3>Top Feature Importances</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={topFeatures} layout="vertical" margin={{ left: 80, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
                <XAxis
                  type="number"
                  stroke="#111827"
                  tick={{ fill: "#111827" }}
                  label={{ value: "Importance", position: "insideBottomRight", dy: 10, fill: "#111827" }}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={220}
                  stroke="#111827"
                  tick={{ fill: "#111827" }}
                />
                <Tooltip />
                <Legend verticalAlign="top" height={24} />
                <Bar
                  dataKey="importance"
                  name="Importance"
                  fill="#38bdf8"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Fail rate by year */}
          <div className="data-card">
            <h3>Fail Rate by Year</h3>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={fail_rate_by_year}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.15)" />
                <XAxis
                  dataKey="year"
                  stroke="#111827"
                  tick={{ fill: "#111827" }}
                />
                <YAxis
                  stroke="#111827"
                  tick={{ fill: "#111827" }}
                  domain={[0, "dataMax"]}
                  tickFormatter={(v) => `${(v * 100).toFixed(1)}%`}
                  label={{ value: "Fail rate", angle: -90, position: "insideLeft", fill: "#111827" }}
                />
                <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
                <Legend verticalAlign="top" height={24} />
                <Bar dataKey="rate" name="Fail rate" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        </div>
      </div>
    </section>

  );
}
