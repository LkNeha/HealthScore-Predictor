import React, { useEffect, useState } from "react";
import CountUp from "react-countup";
import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { fetchCustomerNeighborhoods } from "../services/apiClient";

const SUMMARY_FIELDS = [
  { key: "total_inspections", label: "Total inspections", decimals: 0 },
  { key: "avg_violation_count", label: "Avg violations", decimals: 1 },
  {
    key: "avg_days_since_last_inspection",
    label: "Avg days since last",
    decimals: 0,
  },
  { key: "first_inspection_rate", label: "First inspection rate", decimals: 1, percent: true },
  {
    key: "avg_violation_count_last_3",
    label: "Avg violations (last 3)",
    decimals: 1,
  },
  {
    key: "avg_prev_rating_majority_3",
    label: "Prev rating majority",
    decimals: 1,
  },
];

const DEFAULT_CENTER = [37.773972, -122.431297];

const markerIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

function renderMetricValue(summary, field) {
  const value = summary[field.key];
  if (value === null || value === undefined) {
    return "-";
  }

  const target = field.percent ? Number(value) * 100 : Number(value);
  if (Number.isNaN(target)) {
    return "-";
  }

  return (
    <CountUp
      end={target}
      duration={1}
      decimals={field.decimals || 0}
      suffix={field.percent ? "%" : ""}
    />
  );
}

function formatNumber(value, decimals = 0) {
  if (value === null || value === undefined) {
    return "-";
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return "-";
  }
  return numeric.toFixed(decimals);
}

function formatCount(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return "-";
  }
  return numeric.toLocaleString();
}

function formatDate(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function renderFacilityStatus(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return "-";
  }
  return numeric === 0 ? "Clean" : "Risky";
}

export default function NeighborhoodAnalyzer() {
  const [options, setOptions] = useState([]);
  const [selected, setSelected] = useState("");
  const [summary, setSummary] = useState(null);
  const [topRestaurants, setTopRestaurants] = useState([]);
  const [mapPoints, setMapPoints] = useState([]);
  const [loadingOptions, setLoadingOptions] = useState(true);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [error, setError] = useState(null);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    async function loadOptions() {
      setLoadingOptions(true);
      try {
        const payload = await fetchCustomerNeighborhoods();
        setOptions(payload.options || []);
        setError(null);
      } catch (err) {
        console.error(err);
        setError("Unable to load neighborhood options.");
      } finally {
        setLoadingOptions(false);
      }
    }

    loadOptions();
  }, []);

  const handleSelect = async (evt) => {
    const value = evt.target.value;
    setSelected(value);
    setSummary(null);
    setTopRestaurants([]);
    setMapPoints([]);
    setMessage(null);

    if (!value) {
      return;
    }

    setLoadingSummary(true);
    try {
      const payload = await fetchCustomerNeighborhoods(value);
      setSummary(payload.summary || null);
      setTopRestaurants(payload.top_restaurants || []);
      setMapPoints(payload.map_points || []);
      setMessage(payload.message || null);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Unable to load neighborhood insights.");
    } finally {
      setLoadingSummary(false);
    }
  };

  return (
    <div className="prediction-panel">
      <div className="prediction-header">
        <div>
          <h2 className="title">NEIGHBORHOOD INSIGHTS</h2>
          <p className="prediction-sub">
            Compare aggregated restaurant performance for San Francisco neighborhoods.
          </p>
        </div>
      </div>

      <div className="field" style={{ maxWidth: 420 }}>
        <label>Analyze neighborhood</label>
        <select
          className="select-inspection"
          value={selected}
          onChange={handleSelect}
          disabled={loadingOptions}
        >
          <option value="">Select a neighborhood</option>
          {options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>

      {loadingSummary && (
        <p className="prediction-sub" style={{ marginTop: 14 }}>
          Loading neighborhood metrics...
        </p>
      )}

      {error && (
        <p className="error-text" style={{ marginTop: 14 }}>
          {error}
        </p>
      )}

      {!loadingSummary && message && !summary && (
        <p className="prediction-sub" style={{ marginTop: 14 }}>
          {message}
        </p>
      )}

      {!loadingSummary && selected && (
        <>
          <div className="panel" style={{ marginTop: 16 }}>
            <div className="prediction-header" style={{ marginBottom: 10 }}>
              <div>
                <h3 className="new" style={{ margin: 0 }}>PREFERRED RESTAURANTS</h3>
                {/* <p className="prediction-sub">
                  Ranked using Google prominence and recent inspection cleanliness.
                </p> */}
              </div>
            </div>
            {topRestaurants.length > 0 ? (
              <div className="table-scroll">
                <table className="top-restaurants-table">
                  <thead>
                    <tr>
                      <th>Restaurant</th>
                      <th>Rating</th>
                      <th>Reviews</th>
                      <th>Inspections</th>
                      {/* <th>Avg violations</th> */}
                      <th>Last inspection</th>
                      <th>Facility status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topRestaurants.map((biz) => (
                      <tr key={`${biz.name}-${biz.lat}-${biz.lon}`}>
                        <td>
                          <div className="biz-name">
                            <strong>{biz.name || "Unknown"}</strong>
                            <span>{biz.address || "Address unavailable"}</span>
                          </div>
                        </td>
                        <td>{formatNumber(biz.rating, 1)}</td>
                        <td>{formatCount(biz.user_ratings_total)}</td>
                        <td>{formatCount(biz.total_inspections)}</td>
                        {/* <td>{formatNumber(biz.avg_violation_count, 1)}</td> */}
                        <td>{formatDate(biz.last_inspection_date)}</td>
                        <td>{renderFacilityStatus(biz.facility_rating_status)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="prediction-sub" style={{ marginTop: 8 }}>
                No standout restaurants surfaced for this neighborhood yet.
              </p>
            )}
          </div>

          {mapPoints.length > 0 && (
            <div className="panel map-panel" style={{ marginTop: 16 }}>
              <div className="prediction-header" style={{ marginBottom: 10 }}>
                <div>
                  <h3 className="new" style={{ margin: 0 }}>MAP VIEW</h3>
                  {/* <p className="prediction-sub">
                    Spot the highest-signal venues nearby based on rating, reviews, and cleanliness.
                  </p> */}
                </div>
              </div>
              <NeighborhoodMap points={mapPoints} />
            </div>
          )}
        </>
      )}
    </div>
  );
}

function NeighborhoodMap({ points }) {
  const normalized = points
    .map((pt) => ({
      ...pt,
      lat: typeof pt.lat === "string" ? parseFloat(pt.lat) : pt.lat,
      lon: typeof pt.lon === "string" ? parseFloat(pt.lon) : pt.lon,
    }))
    .filter((pt) => Number.isFinite(pt.lat) && Number.isFinite(pt.lon));

  const centerPoint = normalized.length > 0 ? normalized[0] : null;
  const center = centerPoint ? [centerPoint.lat, centerPoint.lon] : DEFAULT_CENTER;

  if (normalized.length === 0) {
    return (
      <p className="prediction-sub" style={{ marginTop: 8 }}>
        Map data unavailable for this neighborhood.
      </p>
    );
  }

  return (
    <MapContainer center={center} zoom={13} scrollWheelZoom={false} className="neighborhood-map">
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      {normalized.map((point) => (
        <Marker
          key={`${point.name}-${point.lat}-${point.lon}`}
          position={[point.lat, point.lon]}
          icon={markerIcon}
        >
          <Popup>
            <strong>{point.name}</strong>
            <br />
            {point.rating ? `${formatNumber(point.rating, 1)} ★` : "No rating"}
            {point.user_ratings_total ? ` · ${formatCount(point.user_ratings_total)} reviews` : ""}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}
