import React from "react";
import inspectorImg from "../images/inspector.jpg";
import customerImg from "../images/customer.jpg";
import dataImg from "../images/data.jpg";
import futureImg from "../images/future.jpg";
// import "../components/DashboardCards.css";
import "../index.css"

export default function DashboardCards({ onSelect }) {
return (
<div className="card-grid">
  <div
    className="card"
    style={{ backgroundImage: `url(${inspectorImg})` }}
    onClick={() => onSelect("inspector")}
  >
    <div className="card-inner">
      <h2>Inspector</h2>
      <p>Analyze and inspect data patterns</p>
    </div>
  </div>

  <div
    className="card"
    style={{ backgroundImage: `url(${customerImg})` }}
    onClick={() => onSelect("customer")}
  >
    <div className="card-inner">
      <h2>Customer</h2>
      <p>Track customer metrics and engagement</p>
    </div>
  </div>

  <div
    className="card"
    style={{ backgroundImage: `url(${dataImg})` }}
    onClick={() => onSelect("data")}
  >
    <div className="card-inner">
      <h2>Data Insights</h2>
      <p>Discover insights and explore datasets</p>
    </div>
  </div>
  <div
    className="card"
    style={{ backgroundImage: `url(${futureImg})` }}
    onClick={() => onSelect("explore")}
  >
    <div className="card-inner">
      <h2>Future Scope</h2>
      <p>What's next in the list?</p>
    </div>
  </div>
</div>

);
}