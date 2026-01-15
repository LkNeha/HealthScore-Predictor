import React, { useState, useEffect, useRef } from "react";
import "../index.css";
import CountUp from "react-countup";
import { lookupBusinessCustomer } from "../services/apiClient";

export default function CustomerDashboard() {
	const [query, setQuery] = useState("");
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);
	const [result, setResult] = useState(null);
	const resultRef = useRef(null);

	const handleSubmit = async (e) => {
		e.preventDefault();
		const name = query.trim();
		if (!name) return;

		setLoading(true);
		setError(null);
		setResult(null);

		try {
			const data = await lookupBusinessCustomer(name);
			if (data.error) {
				setError(data.error);
			} else {
				setResult(data);
			}
		} catch (err) {
			console.error(err);
			setError("Unable to load business details.");
		} finally {
			setLoading(false);
		}
	};

	// Auto-scroll to the summary cards whenever a new result loads
	useEffect(() => {
		if (resultRef.current && result) {
			try {
				const rect = resultRef.current.getBoundingClientRect();
				const scrollTop = window.scrollY + rect.top - 80; // small offset for header
				window.scrollTo({ top: scrollTop, behavior: "smooth" });
			} catch (e) {
				// fallback: simple scroll into view
				resultRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
			}
		}
	}, [result]);

	return (
		<div className="dashboard">
			<div className="prediction-panel">
				<div className="prediction-header">
					<div>
						<h2 className="title">CUSTOMER VIEW</h2>
						<p className="prediction-sub">
							Look up a restaurant and see its inspection history and Google rating.
						</p>
					</div>
				</div>

				<form className="prediction-grid" onSubmit={handleSubmit}>
					<div className="field" style={{ gridColumn: "span 4" }}>
						<label>Business name</label>
						<input
							placeholder="Start typing a restaurant name"
							value={query}
							onChange={(e) => setQuery(e.target.value)}
						/>
					</div>
					<div className="field" style={{ alignSelf: "flex-end" }}>
						<button
							type="submit"
							className="primary-btn"
							disabled={loading}
						>
							{loading ? "Searching..." : "Search"}
						</button>
					</div>
				</form>

				{error && <p className="error-text" style={{ marginTop: 12 }}>{error}</p>}

				{result && (
					<>
						{/* Row 1: compact summary boxes in one row */}
						<div
							ref={resultRef}
							className="summary-row"
							style={{ marginTop: 14, gridTemplateColumns: "repeat(4, minmax(0, 1fr))" }}
						>
							<div className="summary-card">
								<p className="summary-label">Google rating</p>
								<p className="summary-value">
									{result.google_rating != null ? (
										<>
											<CountUp style={{fontSize:"2.4rem"}}
												end={result.google_rating}
												duration={1.5}
												decimals={1}
											/>
											<span style={{ fontSize: "2.4rem", marginLeft: 4 }}>/ 5</span>
										</>
									) : (
										"-"
									)}
								</p>
								{result.google_reviews != null && (
									<p className="summary-sub">
										Based on <strong>{result.google_reviews}</strong> reviews
									</p>
								)}
							</div>

							<div className="summary-card">
								<p className="summary-label">Total violations in the Past</p>
								<p className="summary-value">
									{result.total_violations != null ? (
											<CountUp style={{fontSize:"2.4rem"}} end={result.total_violations} duration={1.5} />
										) : (
											"-"
										)}
								</p>
							</div>
                            <div className="summary-card">
								<p className="summary-label">Fails reported</p>
								<p className="summary-value">
									{result.fails != null ? (
											<CountUp style={{fontSize:"2.4rem"}} end={result.fails} duration={1.5} />
										) : (
											"-"
										)}
								</p>
							</div>

							<div className="summary-card">
								<p className="summary-label">Fail rate</p>
								<p className="summary-value">
									{result.fail_rate != null ? (
											<CountUp style={{fontSize:"2.4rem"}}
												end={result.fail_rate * 100}
												duration={1.5}
												decimals={1}
												suffix="%"
											/>
										) : (
											"-"
										)}
								</p>
							</div>
						</div>

						{/* Row 2: map with pin at location */}
						{result.latitude != null && result.longitude != null && (
							<div className="panel" style={{ marginTop: 16 }}>
								<p className="prediction-sub" style={{ marginBottom: 8 }}>
									Location
								</p>
								<p className="prediction-sub" style={{ marginBottom: 8 }}>
									{result.business_name}
									{result.address ? ` • ${result.address}` : ""}
								</p>
								<div style={{ width: "100%", height: 260, borderRadius: 12, overflow: "hidden" }}>
									<iframe
										title="Business location map"
										width="100%"
										height="100%"
										style={{ border: 0 }}
										loading="lazy"
										allowFullScreen
										src={`https://www.google.com/maps?q=${result.latitude},${result.longitude}&z=16&output=embed`}
									/>
								</div>
							</div>
						)}
					</>
				)}
			</div>
		</div>
	);
}