// src/components/DataDashboard.jsx
import React, { useEffect, useState, useRef } from "react";
import CountUp from "react-countup";
import "../index.css";
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
  const [selectedModelId, setSelectedModelId] = useState(null);
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
      containerRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [insights]);

  if (loading)
    return (
      <div className="dashboard">
        <p>Loading model insights...</p>
      </div>
    );
  if (error)
    return (
      <div className="dashboard">
        <p>Error: {error}</p>
      </div>
    );
  if (!insights) return null;

  // NEW: global feature-importance + calibration (from /data-insights)
  const featureImportance = insights?.featureImportance ?? [];
  const calibration = insights?.calibrationCurve ?? [];

  // correlations is still the matrix structure coming from backend
  const correlations = insights?.correlations ?? { features: [], values: [] };

  const {
    total_records,
    models = [],
    roc_curves,
    fail_rate_by_year,
    xg_roc_auc_weighted,
    xg_roc_auc_class2,
    rf_roc_auc_class2,
    rf_roc_auc_weighted,
    n_features: overallNFeatures,
  } = insights;

  const effectiveSelectedModelId =
    selectedModelId || (models[0] ? models[0].id : null);

  const selectedModel =
    models.find((m) => m.id === effectiveSelectedModelId) ||
    models[0] ||
    {};

  const {
    name: selectedName,
    roc_auc: modelRocAuc,
    recall_fail,
    n_features,
    feature_importance,
    classification_report,
  } = selectedModel;

  // Build data for ROC chart (XGBoost and RF curve for class 2)
  const rocData =
    roc_curves?.fpr?.map((fprVal, idx) => ({
      fpr: fprVal,
      tpr_xgb: roc_curves.tpr_xgb?.[idx],
      tpr_rf: roc_curves.tpr_rf?.[idx],
    })) || [];

  // Top 10 feature importance for bar chart (current model)
  const topFeatures = (feature_importance || []).slice(0, 10);

  // Correlation heat data
  const corrFeatures = correlations.features || [];
  const corrValues = correlations.values || [];
  const corrData = [];
  corrFeatures.forEach((rowName, i) => {
    corrFeatures.forEach((colName, j) => {
      corrData.push({
        row: rowName,
        col: colName,
        value: corrValues[i]?.[j],
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

  // Extract both XGBoost and RF metrics for comparison
  const xgbClassificationReport = insights.classification_report;
  const rfClassificationReport = insights.rf_classification_report;
  const xgbConfusionMatrix = insights.confusion_matrix;
  const rfConfusionMatrix = insights.rf_confusion_matrix;
  const xgbFeatureImportance = insights.feature_importance || [];
  const rfFeatureImportance = insights.rf_feature_importance || [];
  const xgbMetrics = insights.xgb_metrics || {};
  const rfMetrics = insights.rf_metrics || {};

  // Prepare data for metrics bar chart (precision, recall, f1-score for each class)
  const getBarChartData = (report) => {
    if (!report) return [];
    return Object.keys(report)
      .filter((k) => k.match(/^\d+$/))
      .map((cls) => ({
        class: `Class ${cls}`,
        precision: report[cls].precision,
        recall: report[cls].recall,
        f1: report[cls]["f1-score"],
      }));
  };
  const xgbBarData = getBarChartData(xgbClassificationReport);
  const rfBarData = getBarChartData(rfClassificationReport);

  // Prepare data for metrics table
  const metricsTableRows = [
    {
      metric: "Accuracy",
      xgb: xgbMetrics.accuracy,
      rf: rfMetrics.accuracy,
    },
    {
      metric: "Macro F1",
      xgb: xgbMetrics.macro_f1,
      rf: rfMetrics.macro_f1,
    },
    {
      metric: "Weighted F1",
      xgb: xgbMetrics.weighted_f1,
      rf: rfMetrics.weighted_f1,
    },
    {
      metric: "Macro Precision",
      xgb: xgbMetrics.macro_precision,
      rf: rfMetrics.macro_precision,
    },
    {
      metric: "Weighted Precision",
      xgb: xgbMetrics.weighted_precision,
      rf: rfMetrics.weighted_precision,
    },
    {
      metric: "Macro Recall",
      xgb: xgbMetrics.macro_recall,
      rf: rfMetrics.macro_recall,
    },
    {
      metric: "Weighted Recall",
      xgb: xgbMetrics.weighted_recall,
      rf: rfMetrics.weighted_recall,
    },
  ];

  return (
    <section className="data-section" ref={containerRef}>
      <div className="dashboard data-dashboard">
        <div className="data-panel">
          <h2 className="title">MODEL INSIGHTS</h2>
          {/* Summary strip */}
          <div className="summary-row">
            {/* XGBoost Accuracy Summary Card */}
           

            {/* Existing summary cards */}
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
                  end={overallNFeatures || n_features || 0}
                  duration={2}
                />
              </div>
              <div className="summary-sub">Engineered predictors</div>
            </div>
             <div className="summary-card">
              <div className="summary-label">Accuracy (XGBoost)</div>
              <div className="summary-value">
                <CountUp
                  end={xgbMetrics.accuracy || 0}
                  duration={2}
                  decimals={3}
                />
              </div>
              <div className="summary-sub">Validation accuracy</div>
            </div>

            {/* Random Forest Accuracy Summary Card */}
            <div className="summary-card">
              <div className="summary-label">Accuracy (Random Forest)</div>
              <div className="summary-value">
                <CountUp
                  end={rfMetrics.accuracy || 0}
                  duration={2}
                  decimals={3}
                />
              </div>
              <div className="summary-sub">Validation accuracy</div>
            </div>

            <div className="summary-card">
              <div className="summary-label">ROC‑AUC (XGBoost)</div>
              <div className="summary-value">
                <CountUp
                  end={xg_roc_auc_weighted || 0}
                  duration={2}
                  decimals={3}
                />
              </div>
              <div className="summary-sub">Weighted AUC from saved metrics</div>
            </div>

            <div className="summary-card">
              <div className="summary-label">
                ROC‑AUC (Random forest, weighted)
              </div>
              <div className="summary-value">
                <CountUp
                  end={rf_roc_auc_weighted || 0}
                  duration={2}
                  decimals={3}
                />
              </div>
              <div className="summary-sub">Weighted AUC from saved metrics</div>
            </div>

            <div className="summary-card">
              <div className="summary-label">ROC‑AUC (XGBoost, class 2)</div>
              <div className="summary-value">
                <CountUp
                  end={xg_roc_auc_class2 || 0}
                  duration={2}
                  decimals={3}
                />
              </div>
              <div className="summary-sub">Class 2 (FAIL) ROC‑AUC</div>
            </div>

            <div className="summary-card">
              <div className="summary-label">
                ROC‑AUC (Random forest, class 2)
              </div>
              <div className="summary-value">
                <CountUp
                  end={rf_roc_auc_class2 || 0}
                  duration={2}
                  decimals={3}
                />
              </div>
              <div className="summary-sub">Class 2 (FAIL) ROC‑AUC</div>
            </div>
          </div>

          <div className="data-grid">
            {/* ROC line */}
            <div className="data-card">
              <h3>ROC Curve</h3>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={rocData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(148, 163, 184, 0.15)"
                  />
                  <XAxis
                    dataKey="fpr"
                    stroke="#111827"
                    tick={{ fill: "#111827" }}
                    tickFormatter={(v) => v.toFixed(3)}
                    label={{
                      value: "FPR",
                      position: "insideBottomRight",
                      dy: 10,
                      fill: "#111827",
                    }}
                  />
                  <YAxis
                    stroke="#111827"
                    tick={{ fill: "#111827" }}
                    label={{
                      value: "TPR",
                      angle: -90,
                      position: "insideLeft",
                      fill: "#111827",
                    }}
                  />
                  <Tooltip formatter={(v) => v.toFixed(3)} />
                  <Legend verticalAlign="top" height={54} />
                  <Line
                    type="monotone"
                    dataKey="tpr_xgb"
                    name="XGBoost (Class 2)"
                    stroke="#ef4444"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="tpr_rf"
                    name="Random Forest (Class 2)"
                    stroke="#38bdf8"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Classification Report Comparison Bar Chart */}
            {/* <div className="data-card">
              <h3>Precision, Recall, F1-score by Class</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={xgbBarData} margin={{ left: 40, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="class" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend verticalAlign="top" height={54} />
                  <Bar dataKey="precision" name="XGB Precision" fill="#ef4444" />
                  <Bar dataKey="recall" name="XGB Recall" fill="#f59e42" />
                  <Bar dataKey="f1" name="XGB F1" fill="#22c55e" />
                </BarChart>
              </ResponsiveContainer>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={rfBarData} margin={{ left: 40, right: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="class" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="precision" name="RF Precision" fill="#38bdf8" />
                  <Bar dataKey="recall" name="RF Recall" fill="#818cf8" />
                  <Bar dataKey="f1" name="RF F1" fill="#f472b6" />
                </BarChart>
              </ResponsiveContainer>
            </div> */}

            {/* Confusion Matrices */}
            {/* <div className="data-card" style={{ display: 'flex', gap: 32 }}>
              <div>
                <h3>XGBoost Confusion Matrix</h3>
                {xgbConfusionMatrix ? (
                  <table className="confusion-matrix">
                    <tbody>
                      {xgbConfusionMatrix.map((row, i) => (
                        <tr key={i}>
                          {row.map((val, j) => (
                            <td key={j}>{val}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p>No data</p>
                )}
              </div>
              <div>
                <h3>Random Forest Confusion Matrix</h3>
                {rfConfusionMatrix ? (
                  <table className="confusion-matrix">
                    <tbody>
                      {rfConfusionMatrix.map((row, i) => (
                        <tr key={i}>
                          {row.map((val, j) => (
                            <td key={j}>{val}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p>No data</p>
                )}
              </div>
            </div> */}

            {/* Feature Importance Comparison */}
            {/* <div className="data-card" style={{ display: 'flex', gap: 32 }}>
              <div style={{ flex: 1 }}>
                <h3>XGBoost Feature Importance</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={xgbFeatureImportance} layout="vertical" margin={{ left: 80, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="feature" width={180} />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div style={{ flex: 1 }}>
                <h3>Random Forest Feature Importance</h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={rfFeatureImportance} layout="vertical" margin={{ left: 80, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="feature" width={180} />
                    <Tooltip />
                    <Bar dataKey="importance" fill="#38bdf8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>  */}

            {/* Model Metrics Table */}
            {/* <div className="data-card">
              <h3>Model Performance Comparison</h3>
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>XGBoost</th>
                    <th>Random Forest</th>
                  </tr>
                </thead>
                <tbody>
                  {metricsTableRows.map((row) => (
                    <tr key={row.metric}>
                      <td>{row.metric}</td>
                      <td>{row.xgb !== undefined ? row.xgb.toFixed(3) : '-'}</td>
                      <td>{row.rf !== undefined ? row.rf.toFixed(3) : '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div> */}
          </div>
        </div>
      </div>
    </section>
  );
}
