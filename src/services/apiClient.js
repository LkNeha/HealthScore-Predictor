// import axios from "axios";

const BASE_URL = "http://localhost:8000";

export async function runPrediction(payload) {
  const res = await fetch(`${BASE_URL}/predict-inspector`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`Backend error: ${res.status}`);
  }
  return res.json(); // { risk_label, risk_score }
}

export async function lookupBusinessCustomer(name) {
  const res = await fetch(
    `${BASE_URL}/customer-business?name=${encodeURIComponent(name)}`
  );

  if (!res.ok) {
    throw new Error(`Backend error: ${res.status}`);
  }

  return res.json();
}
export async function fetchCustomerNeighborhoods(name) {
  const endpoint = name
    ? `${BASE_URL}/customer-neighborhoods?name=${encodeURIComponent(name)}`
    : `${BASE_URL}/customer-neighborhoods`;

  const res = await fetch(endpoint);
  if (!res.ok) {
    throw new Error(`Backend error: ${res.status}`);
  }

  return res.json();
}

export async function fetchCustomerNeighborhoods(name) {
  const endpoint = name
    ? `${BASE_URL}/customer-neighborhoods?name=${encodeURIComponent(name)}`
    : `${BASE_URL}/customer-neighborhoods`;

  const res = await fetch(endpoint);
  if (!res.ok) {
    throw new Error(`Backend error: ${res.status}`);
  }

  return res.json();
}


// example: GET /api/inspections
// export async function getInspections(params) {
//   const res = await axios.get(`${API_BASE}/api/inspections`, { params });
//   return res.data;
// }

// // example: POST /api/predict
// export async function predictFailure(payload) {
//   const res = await axios.post(`${API_BASE}/api/predict`, payload);
//   return res.data;
// }

// // example: GET /api/metrics
// export async function getMetrics() {
//   const res = await axios.get(`${API_BASE}/api/metrics`);
//   return res.data;
// }