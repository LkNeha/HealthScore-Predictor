import React, { useState } from "react";
import DashboardCards from "../components/DashboardCards";
import InspectorDashboard from "../components/InspectorDashboard";
import CustomerDashboard from "../components/CustomerDashboard";
import ExploreDashboard from "../components/ExploreDashboard";
import DataDashboard from "../components/DataDashboard";
import "../pages/AnalyticsDashboard.css";
import CountUp from "react-countup";

import img1 from "../images/final.jpg";

// const heroImages=[img1, img2, img3, img4, img5];

export default function AnalyticsDashboard() {
    const [activeDashboard, setActiveDashboard] = useState(null);
    const [dashboardClicks, setDashboardClicks] = useState({
        inspector: 0,
        customer: 0,
        explore: 0,
    });
    const handleSelect = (type) => {
        setActiveDashboard(type);
        setDashboardClicks((prev) => ({
            ...prev,
            [type]: prev[type] + 1,   // increment on every click
        }));
    };
    return (
        <div className="page">

            <section className="hero-slider-loop">
                <div
                    className="hero-bg"
                    style={{ backgroundImage: `url(${img1})` }}
                />
                <div className="hero-text">
                    <h1>San Francisco Health Inspection 2026</h1>
                    <p>Data‑driven insights for safer, smarter kitchens.</p>
                </div>
            </section>
            <section className="dashboard-area">
                <div className="container">
                    <h2 className="titleAnalytics">ANALYTICS DASHBOARD</h2>

                    <div className="dash-row">
                        {/* left: cards (unchanged markup inside) */}
                        <div className="dash-left">
                            <DashboardCards onSelect={handleSelect} />
                        </div>

                        {/* right: text */}
                        <div className="dash-right dash-right-metrics">
                            <h3 className="dash-kicker">TODAY’S HIGHLIGHTS</h3>

                            <div className="metric-block">
                                <div className="metric-value"><CountUp end={32097} duration={2} /></div>
                                <div className="metric-label">inspections completed</div>
                            </div>

                            <div className="metric-block">
                                <div className="metric-value"><CountUp end={7.1} duration={2} suffix="%" /></div>
                                <div className="metric-label">facilities compliant</div>
                            </div>

                            <div className="metric-block">
                                <div className="metric-value">
                                    <CountUp end={1453} duration={2} />
                                </div>
                                <div className="metric-label">high‑risk alerts flagged</div>
                            </div>
                        </div>

                    </div>

                    <div className="dashboard-detail">
                        {activeDashboard === "inspector" && (
                            <InspectorDashboard
                                key={dashboardClicks.inspector}   // forces remount
                                onBack={() => setActiveDashboard(null)}
                            />
                        )}

                        {activeDashboard === "customer" && (
                            <CustomerDashboard
                                key={dashboardClicks.customer}
                            />
                        )}
                        {activeDashboard === "data" && (
                            <DataDashboard
                                key={dashboardClicks.data}
                            />
                        )}

                        {activeDashboard === "explore" && (
                            <ExploreDashboard
                                key={dashboardClicks.explore}
                            />
                        )}
                    </div>
                </div>
            </section>



        </div>
    );
}
