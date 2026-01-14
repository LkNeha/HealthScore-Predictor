import React, { useState, useEffect, useRef } from "react";
import CustomerDashboard from "./CustomerDashboard";
import NeighborhoodAnalyzer from "./NeighborhoodAnalyzer";

const MODES = {
  restaurant: {
    label: "Restaurant Lookup",
    helper: "Search for a single business and see its inspection history.",
  },
  neighborhood: {
    label: "Analyze Neighborhood",
    helper: "Explore aggregated inspection trends across San Francisco neighborhoods.",
  },
};

export default function CustomerPortal() {
  const [mode, setMode] = useState(null);
  const [chooserOpen, setChooserOpen] = useState(true);
  const restaurantRef = useRef(null);
  const neighborhoodRef = useRef(null);

  const handleSelect = (value) => {
    setMode(value);
    setChooserOpen(false);
  };

  useEffect(() => {
    if (chooserOpen || !mode) {
      return;
    }
    const targetRef = mode === "restaurant" ? restaurantRef : neighborhoodRef;
    if (targetRef.current) {
      window.requestAnimationFrame(() => {
        targetRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  }, [chooserOpen, mode]);

  return (
    <div className="customer-portal">
      {chooserOpen && (
        <div className="customer-mode-overlay">
          <div className="customer-mode-card">
            <h3>How would you like to explore?</h3>
            <div className="customer-mode-options">
              <button
                className="customer-mode-btn"
                onClick={() => handleSelect("restaurant")}
              >
                <span style={{ display: "block", textAlign: "center" }}>Restaurant view</span>
                
              </button>
              <button
                className="customer-mode-btn secondary"
                onClick={() => handleSelect("neighborhood")}
              >
                <span style={{ display: "block", textAlign: "center" }}>Analyze neighborhood</span>
                
              </button>
            </div>
          </div>
        </div>
      )}

      {!chooserOpen && mode === "restaurant" && (
        <div ref={restaurantRef}>
          <CustomerDashboard />
        </div>
      )}
      {!chooserOpen && mode === "neighborhood" && (
        <div ref={neighborhoodRef}>
          <NeighborhoodAnalyzer />
        </div>
      )}
    </div>
  );
}
