from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import math
import numpy as np
import pandas as pd
app = FastAPI()

# ---- CORS so React (localhost:3000) can call this ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- load model once ----
# MODEL_PATH = Path(__file__).parent / "model" / "xgboost_tuned_scaleweight.pkl"
MODEL_PATH='C:/Users/lkneh/HealthScore-Predictor/backend/model/xgboost_tuned_scaleweight.pkl'  # adjust as needed
raw_obj = joblib.load(MODEL_PATH)
# if you know you saved {"model": xgb_model, ...}
if isinstance(raw_obj, dict) and "model" in raw_obj:
    model = raw_obj["model"]
    scaler = raw_obj.get("scaler", None)
    feature_names = raw_obj.get("feature_names")
    X_val = raw_obj.get("X_val")
    y_val = raw_obj.get("y_val")
else:
    model = raw_obj
    scaler = None
    feature_names = None
    X_val = None
    y_val = None

# ---- feature order expected by model ----
MODEL_FEATURES = [
    "latitude",
    "longitude",
    "avg_violations_last_3",
    "fail_rate_last_3",
    "days_since_last_inspection",
    "trend_last_3",
    "BusinessName_id",
    "Address_id",
    "insp_year",
    "insp_month",
    "insp_day",
    "insp_dow",
    "insp_days_since_ref",
    "inspection_type_clean_change of ownership",
    "inspection_type_clean_complaint",
    "inspection_type_clean_complaint (i)",
    "inspection_type_clean_complaint (r)",
    "inspection_type_clean_complaint reinspection/follow-up",
    "inspection_type_clean_foodborne illness",
    "inspection_type_clean_foodborne illness investigation",
    "inspection_type_clean_new construction",
    "inspection_type_clean_new ownership",
    "inspection_type_clean_new ownership (i)",
    "inspection_type_clean_new ownership (r)",
    "inspection_type_clean_new ownership - followup",
    "inspection_type_clean_non-inspection site visit",
    "inspection_type_clean_plan check",
    "inspection_type_clean_plan check (i)",
    "inspection_type_clean_plan check (r)",
    "inspection_type_clean_reinspection",
    "inspection_type_clean_reinspection/followup",
    "inspection_type_clean_routine",
    "inspection_type_clean_site visit",
    "inspection_type_clean_structural",
    "inspection_type_clean_structural inspection",
    "inspection_type_clean_nan",
]

INSPECTION_OHE_FEATURES = [
    f for f in MODEL_FEATURES if f.startswith("inspection_type_clean_")
]

# ---- request / response models ----
class PredictionInput(BaseModel):
    latitude: float
    longitude: float
    avg_violations_last_3: float
    fail_rate_last_3: float
    days_since_last_inspection: int
    trend_last_3: float
    insp_year: int
    insp_month: int
    insp_day: int
    insp_dow: int
    insp_days_since_ref: int
    inspection_type: str   # e.g. "routine", "complaint", "plan check (i)"


class PredictionOutput(BaseModel):
    risk_label: str
    risk_score: float
    outcome_label: int    # 0 or 1
    outcome_text: str 


def make_features(data: PredictionInput) -> np.ndarray:
    feats = {name: 0.0 for name in MODEL_FEATURES}

    feats["latitude"] = data.latitude
    feats["longitude"] = data.longitude
    feats["avg_violations_last_3"] = data.avg_violations_last_3
    feats["fail_rate_last_3"] = data.fail_rate_last_3
    feats["days_since_last_inspection"] = float(data.days_since_last_inspection)
    feats["trend_last_3"] = data.trend_last_3
    feats["insp_year"] = float(data.insp_year)
    feats["insp_month"] = float(data.insp_month)
    feats["insp_day"] = float(data.insp_day)
    feats["insp_dow"] = float(data.insp_dow)
    feats["insp_days_since_ref"] = float(data.insp_days_since_ref)

    # IDs not provided by frontend
    feats["BusinessName_id"] = 0.0
    feats["Address_id"] = 0.0

    col = f"inspection_type_clean_{data.inspection_type}"
    if col in INSPECTION_OHE_FEATURES:
        feats[col] = 1.0
    else:
        feats["inspection_type_clean_nan"] = 1.0

    arr = np.array([feats[name] for name in MODEL_FEATURES], dtype=float)
    return arr.reshape(1, -1)

@app.get("/test-model")
def test_model():
    # build a dummy X with correct shape
    import numpy as np
    X = np.zeros((1, len(MODEL_FEATURES)))
    try:
        p = float(model.predict_proba(X)[0, 1])
        return {"ok": True, "proba": p}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    X = make_features(input_data)
    proba = float(model.predict_proba(X)[0, 1])  # adjust if not binary
    p_fail = float(model.predict_proba(X)[0, 1])  # assuming classes_ == [0,1] and 1 == fail
    y_hat = int(model.predict(X)[0])
    print(type(raw_obj), getattr(raw_obj, "keys", lambda: None)())

    # <0.4 = LOW, 0.4–0.7 = MEDIUM, >0.7 = HIGH.
    if proba > 0.7:
        label = "HIGH"
    elif proba >= 0.4:
        label = "MEDIUM"
    else:
        label = "LOW"
    outcome_text = "FAIL" if y_hat == 1 else "PASS"
    return PredictionOutput(risk_label=label, risk_score=proba,outcome_label=y_hat,outcome_text=outcome_text,)
def safe_float(x):
    # convert to 0.0 
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        if math.isfinite(x):
            return float(x)
        return 0.0   # or None
    return float(x)


def to_native(val):
    """Convert NumPy / pandas scalar types to plain Python for JSON.

    FastAPI's jsonable_encoder can struggle with some NumPy scalars,
    so we normalize anything we know here.
    """
    # NumPy generic scalars (int64, float64, bool_, etc.)
    if isinstance(val, np.generic):
        val = val.item()

    # pandas Timestamp
    try:
        import pandas as _pd  # local import to avoid cycles

        if isinstance(val, _pd.Timestamp):
            return val.isoformat()
    except Exception:
        pass

    # Normalize floats so NaN/Inf don't break JSON
    if isinstance(val, float):
        if not math.isfinite(val):
            return 0.0

    return val


# ---------- Customer view helpers (join inspections with Google Places) ----------
_customer_join_cache = None


def load_customer_join_df():
    """Join encoded inspections with Google Places export on name + lat/lon.

    Cached in memory for fast customer lookups.
    """
    global _customer_join_cache
    if _customer_join_cache is not None:
        return _customer_join_cache

    # paths mirror those used elsewhere in this project
    insp_path = "C:/Users/lkneh/HealthScore-Predictor/data/clean/encoded/HealthInspectionsAll.csv"
    google_path = "C:/Users/lkneh/HealthScore-Predictor/data/raw/sf_restaurants_google.csv"

    try:
        insp = pd.read_csv(insp_path)
        google = pd.read_csv(google_path)
    except Exception as e:
        print("Error loading customer join sources", e)
        _customer_join_cache = pd.DataFrame()
        return _customer_join_cache

    # normalise join keys
    insp = insp.copy()
    google = google.copy()

    insp["name_key"] = insp["BusinessName"].astype(str).str.strip().str.upper()
    google["name_key"] = google["name"].astype(str).str.strip().str.upper()

    # Join primarily on business name key; lat/lon are kept for reference but
    # not required to match exactly (geocoding often differs slightly).
    merged = pd.merge(
        insp,
        google,
        on=["name_key"],
        how="inner",
    )

    if merged.empty:
        _customer_join_cache = merged
        return _customer_join_cache

    # Group by business identity (name/address/rating) and aggregate
    # inspections, then compute a representative lat/lon.
    group_cols = [
        "name_key",
        "BusinessName",
        "address",  # from Google CSV
        "rating",
        "user_ratings_total",
    ]

    agg = (
        merged.groupby(group_cols)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            total_violations=("violation_count", "sum"),
            total_inspections=("failFlag", "size"),
            fails=("failFlag", "sum"),
        )
        .reset_index()
    )

    # compute fail rate safely
    agg["fail_rate"] = agg.apply(
        lambda r: float(r["fails"]) / float(r["total_inspections"]) if r["total_inspections"] > 0 else 0.0,
        axis=1,
    )

    # cache and (optionally) write debug join output
    _customer_join_cache = agg
    try:
        _customer_join_cache.to_csv("customer_join_debug.csv", index=False)
    except Exception:
        # ignore file write issues in production path
        pass
    return _customer_join_cache

@app.get("/data-insights")
def data_insights():
    X_val = raw_obj.get("X_val")
    y_val = raw_obj.get("y_val")
    scaler = raw_obj.get("scaler")
    feature_names = raw_obj.get("feature_names")

    if X_val is None or y_val is None:
        return {"error": "X_val / y_val not in model package"}

    Xv = scaler.transform(X_val) if scaler is not None else X_val
    proba = model.predict_proba(Xv)[:, 1]
    y_pred = model.predict(Xv)

    # AUC, ROC
    auc = safe_float(roc_auc_score(y_val, proba))
    fpr, tpr, thresholds = roc_curve(y_val, proba)
    roc_curve_payload = {
        "fpr": [safe_float(v) for v in fpr],
        "tpr": [safe_float(v) for v in tpr],
        "thresholds": [safe_float(v) for v in thresholds],
    }

    # classification report, clean NaNs
    report_raw = classification_report(y_val, y_pred, output_dict=True)
    clean_report = {}
    for key, val in report_raw.items():
        if isinstance(val, dict):
            clean_report[key] = {k: safe_float(v) for k, v in val.items()}
        else:
            clean_report[key] = safe_float(val)

    recall_fail = safe_float(report_raw["1"]["recall"])

    # feature importance
    importances = getattr(model, "feature_importances_", None)
    if importances is not None and feature_names is not None:
        # Aggregate one-hot inspection_type_clean_* columns into a single
        # logical feature so they don't dominate the top-importance list.
        agg_importance = {}
        n_inspection_type_feats = 0
        for f, i in zip(feature_names, importances):
            imp_val = safe_float(i)
            if f.startswith("inspection_type_clean"):
                key = "inspection_type"
                n_inspection_type_feats += 1
            else:
                key = f
            agg_importance[key] = agg_importance.get(key, 0.0) + (imp_val or 0.0)

        # Optionally scale the aggregated inspection_type importance
        # so that it's comparable to a single feature instead of the
        # sum over all one-hot columns (similar to dividing by 10).
        if "inspection_type" in agg_importance and n_inspection_type_feats > 0:
            agg_importance["inspection_type"] /= float(n_inspection_type_feats)

        feat_imp = [
            {"feature": f, "importance": safe_float(imp)}
            for f, imp in sorted(
                agg_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
    else:
        feat_imp = None

    # correlations
    corr_features = [
        "failFlag",
        "avg_violations_last_3",
        "fail_rate_last_3",
        "days_since_last_inspection",
        "trend_last_3",
    ]
    df_val = pd.concat(
        [X_val[corr_features[1:]], y_val.rename("failFlag")], axis=1
    )
    corr = df_val.corr()
    corr_matrix = {
        "features": corr.columns.tolist(),
        "values": [[safe_float(v) for v in row] for row in corr.values],
    }

    # fail rate by year
    if "insp_year" in X_val.columns:
        df_year = pd.concat(
            [X_val["insp_year"], y_val.rename("failFlag")], axis=1
        )
        grouped = (
            df_year.groupby("insp_year")["failFlag"]
            .mean()
            .reset_index()
            .rename(columns={"failFlag": "fail_rate"})
        )
        fail_rate_year = [
            {"year": int(r["insp_year"]), "rate": safe_float(r["fail_rate"])}
            for _, r in grouped.iterrows()
        ]
    else:
        fail_rate_year = []

    return {
        "roc_auc": auc,
        "recall_fail": recall_fail,
        "total_records": int(len(X_val) + len(y_val)),
        "n_features": len(feature_names) if feature_names is not None else None,
        "roc_curve": roc_curve_payload,
        "classification_report": clean_report,
        "feature_importance": feat_imp,
        "correlations": corr_matrix,
        "fail_rate_by_year": fail_rate_year,
    }
import pandas as pd


@app.get("/inspector-insights")
def inspector_insights():
    """Inspector risk analysis dashboard payload based on HealthInspectionsAll.csv.

    Returns:
        - schema: column names, dtypes, and an example value
        - preview_rows: a few sample rows for table preview
        - summary: total inspections, fails, fail rate, risk-band counts
        - inspection_trends: yearly totals / fails / fail_rate
        - inspector_comparison: per BusinessName_id stats (top 10)
        - risk_distribution: counts by voilation_count (if present)
        - high_risk_records: top high-risk businesses (if columns present)
    """

    # load the encoded CSV directly
    df = pd.read_csv(
        "C:/Users/lkneh/HealthScore-Predictor/data/clean/encoded/HealthInspectionsAll.csv"
    )

    # --- schema + preview ---
    schema = []
    for col in df.columns:
        col_series = df[col]
        # pick first non-null example if available
        example_val = None
        non_null = col_series.dropna()
        if len(non_null) > 0:
            example_val = to_native(non_null.iloc[0])
        schema.append(
            {
                "name": col,
                "dtype": str(col_series.dtype),
                "example": example_val,
            }
        )

    # ensure preview rows also only contain native Python types
    preview_raw = df.head(5).to_dict(orient="records")
    preview_rows = [
        {k: to_native(v) for k, v in row.items()}
        for row in preview_raw
    ]

    # --- summary stats ---
    total_inspections = int(len(df))
    if "failFlag" in df.columns:
        total_fails = int(df["failFlag"].sum())
        overall_fail_rate = float(total_fails / total_inspections) if total_inspections > 0 else 0.0
    else:
        total_fails = None
        overall_fail_rate = None

    # rating breakdown (use existing facility_rating_status instead of synthetic bands)
    risk_band_counts = []
    if "facility_rating_status" in df.columns:
        band_counts = (
            df.groupby("facility_rating_status")[df.columns[0]]
            .size()
            .reset_index(name="count")
        )
        risk_band_counts = [
            {"band": str(r["facility_rating_status"]), "count": int(r["count"])}
            for _, r in band_counts.iterrows()
        ]

    # high/low-risk restaurant counts at business level
    high_restaurants_count = None
    low_restaurants_count = None
    business_risk_bands = []
    if "BusinessName" in df.columns and "failFlag" in df.columns:
        biz_fail = (
            df.groupby("BusinessName")["failFlag"]
            .agg(total_inspections="size", fails="sum")
            .reset_index()
        )
        high_restaurants_count = int((biz_fail["fails"] >= 1).sum())
        low_restaurants_count = int((biz_fail["fails"] == 0).sum())

        # derive simple business-level risk bands inspectors can scan quickly
        biz_fail["fail_rate"] = biz_fail["fails"] / biz_fail["total_inspections"]

        def _band(row):
            if row["fails"] == 0:
                return "Low (0 fails)"
            if row["fail_rate"] >= 0.5:
                return "High (>=50% fails)"
            return "Medium"

        biz_fail["risk_band"] = biz_fail.apply(_band, axis=1)

        band_counts = (
            biz_fail.groupby("risk_band")["BusinessName"]
            .size()
            .reset_index(name="businesses")
        )

        business_risk_bands = [
            {
                "band": str(to_native(r.risk_band)),
                "businesses": int(to_native(r.businesses)),
            }
            for _, r in band_counts.iterrows()
        ]

    summary = {
        "total_inspections": total_inspections,
        "total_fails": total_fails,
        "overall_fail_rate": overall_fail_rate,
        "risk_band_counts": risk_band_counts,
        "business_risk_bands": business_risk_bands,
        "high_risk_restaurants": high_restaurants_count,
        "low_risk_restaurants": low_restaurants_count,
    }

    # --- inspection trends over time (per year) ---
    inspection_trends = []
    if "insp_year" in df.columns and "failFlag" in df.columns:
        trend = (
            df.groupby("insp_year")["failFlag"]
            .agg(total="size", fails="sum")
            .reset_index()
        )
        trend["passes"] = trend["total"] - trend["fails"]
        trend["fail_rate"] = trend["fails"] / trend["total"]
        inspection_trends = [
            {
                "year": int(to_native(r.insp_year)),
                "total": int(to_native(r.total)),
                "fails": int(to_native(r.fails)),
                "passes": int(to_native(r.passes)),
                "fail_rate": float(to_native(r.fail_rate)),
            }
            for _, r in trend.iterrows()
        ]

    # --- inspection type fail rates (which types are riskier?) ---
    inspection_type_stats = []
    if "inspection_type" in df.columns and "failFlag" in df.columns:
        type_g = (
            df.groupby("inspection_type")["failFlag"]
            .agg(total="size", fails="sum")
            .reset_index()
        )
        type_g["fail_rate"] = type_g["fails"] / type_g["total"]

        # focus on the most common types so the chart stays readable
        type_g = type_g.sort_values("total", ascending=False).head(10)

        inspection_type_stats = [
            {
                "inspection_type": str(to_native(r.inspection_type)),
                "total": int(to_native(r.total)),
                "fails": int(to_native(r.fails)),
                "fail_rate": float(to_native(r.fail_rate)),
            }
            for _, r in type_g.iterrows()
        ]

    # --- fail rates by day of week ---
    inspection_dow_stats = []
    if "insp_dow" in df.columns and "failFlag" in df.columns:
        dow_df = df[["insp_dow", "failFlag"]].copy()
        dow_df["insp_dow"] = pd.to_numeric(dow_df["insp_dow"], errors="coerce")
        dow_df = dow_df.dropna(subset=["insp_dow"]).astype({"insp_dow": int})

        dow_g = (
            dow_df.groupby("insp_dow")["failFlag"]
            .agg(total="size", fails="sum")
            .reset_index()
        )
        dow_g["fail_rate"] = dow_g["fails"] / dow_g["total"]

        dow_labels = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
        inspection_dow_stats = [
            {
                "dow_index": int(to_native(r.insp_dow)),
                "dow": dow_labels.get(int(to_native(r.insp_dow)), str(to_native(r.insp_dow))),
                "total": int(to_native(r.total)),
                "fails": int(to_native(r.fails)),
                "fail_rate": float(to_native(r.fail_rate)),
            }
            for _, r in dow_g.iterrows()
        ]

    # --- risk vs time since last inspection (bucketed days) ---
    inspection_delay_stats = []
    if {"days_since_last_inspection", "failFlag"}.issubset(df.columns):
        delay_df = df[["days_since_last_inspection", "failFlag"]].copy()
        # ensure numeric days, drop rows we can't interpret
        delay_df["days_since_last_inspection"] = pd.to_numeric(
            delay_df["days_since_last_inspection"], errors="coerce"
        )
        delay_df = delay_df.dropna(subset=["days_since_last_inspection"])

        bins = [-1, 0, 30, 90, 180, 365, float("inf")]
        labels = ["0", "1–30", "31–90", "91–180", "181–365", "366+"]

        delay_df["bucket"] = pd.cut(
            delay_df["days_since_last_inspection"], bins=bins, labels=labels
        )

        d_g = (
            delay_df.groupby("bucket")["failFlag"]
            .agg(total="size", fails="sum")
            .reset_index()
        )
        d_g["fail_rate"] = d_g["fails"] / d_g["total"]

        inspection_delay_stats = [
            {
                "bucket": str(to_native(r.bucket)),
                "total": int(to_native(r.total)),
                "fails": int(to_native(r.fails)),
                "fail_rate": float(to_native(r.fail_rate)),
            }
            for _, r in d_g.iterrows()
        ]

    # --- calendar-style heatmap: insp_dow x insp_month ---
    dow_month_heatmap = []
    if {"insp_month", "insp_dow", "failFlag"}.issubset(df.columns):
        cal_cols = ["insp_month", "insp_dow", "failFlag"]
        if "violation_count" in df.columns:
            cal_cols.append("violation_count")
        cal_df = df[cal_cols].copy()
        cal_df["insp_month"] = pd.to_numeric(cal_df["insp_month"], errors="coerce")
        cal_df["insp_dow"] = pd.to_numeric(cal_df["insp_dow"], errors="coerce")
        cal_df = cal_df.dropna(subset=["insp_month", "insp_dow"])
        cal_df = cal_df.astype({"insp_month": int, "insp_dow": int})

        group_keys = ["insp_month", "insp_dow"]
        agg_dict = {"failFlag": ["size", "mean"]}
        if "violation_count" in cal_df.columns:
            agg_dict["violation_count"] = "mean"

        cal_g = cal_df.groupby(group_keys).agg(agg_dict).reset_index()
        cal_g.columns = [
            "insp_month",
            "insp_dow",
            "total",
            "fail_rate",
        ] + (["avg_violations"] if "violation_count" in cal_df.columns else [])

        month_labels = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        dow_labels = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}

        for _, r in cal_g.iterrows():
            row = {
                "month_index": int(to_native(r.insp_month)),
                "month": month_labels.get(int(to_native(r.insp_month)), str(to_native(r.insp_month))),
                "dow_index": int(to_native(r.insp_dow)),
                "dow": dow_labels.get(int(to_native(r.insp_dow)), str(to_native(r.insp_dow))),
                "total": int(to_native(r.total)),
                "fail_rate": float(to_native(r.fail_rate)),
            }
            if "avg_violations" in cal_g.columns:
                row["avg_violations"] = float(to_native(r.avg_violations))
            dow_month_heatmap.append(row)

    # --- inspector / business comparison (top 10)
    # kept for backwards compatibility, but no longer charted on the UI
    inspector_comparison = []
    if "BusinessName_id" in df.columns and "failFlag" in df.columns:
        comp = (
            df.groupby("BusinessName_id")["failFlag"]
            .agg(total="size", fails="sum")
            .reset_index()
        )
        comp["fail_rate"] = comp["fails"] / comp["total"]
        comp = comp.sort_values("total", ascending=False).head(10)
        inspector_comparison = [
            {
                "Business": int(to_native(r.BusinessName_id)),
                "total": int(to_native(r.total)),
                "fails": int(to_native(r.fails)),
                "fail_rate": float(to_native(r.fail_rate)),
            }
            for _, r in comp.iterrows()
        ]

    # --- neighbourhood analysis using latitude / longitude ---
    neighbourhood_analysis = []
    if {"latitude", "longitude", "failFlag"}.issubset(df.columns):
        loc = df[["latitude", "longitude", "failFlag"]].dropna()

        # bucket into approximate neighbourhoods by rounding lat/lon
        loc["lat_bin"] = loc["latitude"].round(3)
        loc["lon_bin"] = loc["longitude"].round(3)

        neigh = (
            loc.groupby(["lat_bin", "lon_bin"])["failFlag"]
            .agg(total="size", fails="sum")
            .reset_index()
        )
        neigh["fail_rate"] = neigh["fails"] / neigh["total"]

        # limit to most frequently inspected neighbourhoods so the chart stays readable
        neigh = neigh.sort_values("total", ascending=False).head(60)

        neighbourhood_analysis = [
            {
                "lat": float(to_native(r.lat_bin)),
                "lon": float(to_native(r.lon_bin)),
                "total": int(to_native(r.total)),
                "fails": int(to_native(r.fails)),
                "fail_rate": float(to_native(r.fail_rate)),
            }
            for _, r in neigh.iterrows()
        ]

    inspection_year_pie = []

    if "insp_year" in df.columns:
        # Simple value_counts by year to avoid fragile column naming
        year_counts = df["insp_year"].value_counts().sort_index()

        inspection_year_pie = [
            {
                "label": str(int(to_native(year))),
                "value": int(to_native(count)),
            }
            for year, count in year_counts.items()
        ]

    
    high_risk_records = []

    required_cols = {"BusinessName", "insp_year", "failFlag", "violation_count"}
    if required_cols.issubset(df.columns):
        high_risk = (
            df.groupby(["BusinessName", "insp_year"])
            .agg(
                total_inspections=("failFlag", "size"),
                fails=("failFlag", "sum"),
                avg_violations=("violation_count", "mean"),
            )
            .reset_index()
        )

        high_risk["fail_rate"] = high_risk["fails"] / high_risk["total_inspections"]

        # define "high risk": at least one failure, then sort by fail_rate and avg_violations
        high_risk = high_risk[high_risk["fails"] >= 1]

        high_risk = high_risk.sort_values(
            ["fail_rate", "avg_violations"],
            ascending=False
        ).head(30)

        high_risk_records = [
            {
                "BusinessName": to_native(r.BusinessName),
                "year": int(to_native(r.insp_year)),
                "total_inspections": int(to_native(r.total_inspections)),
                "fails": int(to_native(r.fails)),
                "fail_rate": float(to_native(r.fail_rate)),
                "avg_violations": float(to_native(r.avg_violations)),
            }
            for _, r in high_risk.iterrows()
        ]
    violation_severity_pie = []

    if "violation_count" in df.columns:
        bins = [-1, 0, 2, 5, float("inf")]
        labels = ["0", "1–2", "3–5", "6+"]

        severity = pd.cut(df["violation_count"], bins=bins, labels=labels)
        severity_counts = severity.value_counts().sort_index()

        violation_severity_pie = [
            {
                "label": str(label),
                "value": int(count),
            }
            for label, count in severity_counts.items()
        ]

    return {
        "summary": summary,
        "inspection_trends": inspection_trends,
        "inspection_type_stats": inspection_type_stats,
        "inspection_delay_stats": inspection_delay_stats,
        "inspection_dow_stats": inspection_dow_stats,
        "dow_month_heatmap": dow_month_heatmap,
        "inspector_comparison": inspector_comparison,
        "neighbourhood_analysis": neighbourhood_analysis,
        "inspection_year_pie": inspection_year_pie,
        "high_risk_records": high_risk_records,
        "violation_severity_pie": violation_severity_pie,
    }


@app.get("/customer-business")
def customer_business(name: str):
    """Lookup a single business for the customer dashboard.

    Joins inspection history with Google rating using
    name + latitude + longitude as keys.
    """
    df = load_customer_join_df()
    if df is None or df.empty:
        return {"error": "Joined business data not available"}

    key = name.strip().upper()
    if not key:
        return {"error": "Business name is required"}

    exact = df[df["name_key"] == key]
    if not exact.empty:
        candidates = exact
    else:
        # fall back to contains search for partial matches
        candidates = df[df["name_key"].str.contains(key, na=False)]

    if candidates.empty:
        return {"error": "No matching business found"}

    # pick the most inspected match as the best candidate
    row = candidates.sort_values("total_inspections", ascending=False).iloc[0]

    resp = {
        "business_name": to_native(row["BusinessName"]),
        "address": to_native(row["address"]),
        "latitude": safe_float(row["latitude"]),
        "longitude": safe_float(row["longitude"]),
        "total_inspections": int(row["total_inspections"]),
        "total_violations": int(row["total_violations"]),
        "fails": int(row["fails"]),
        "fail_rate": safe_float(row["fail_rate"]),
        "google_rating": safe_float(row["rating"]),
    }

    reviews = row.get("user_ratings_total")
    try:
        resp["google_reviews"] = int(reviews) if pd.notna(reviews) else None
    except Exception:
        resp["google_reviews"] = None

    return resp
