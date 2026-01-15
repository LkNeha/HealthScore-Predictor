from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import math
import numpy as np
import pandas as pd
from datetime import date
from typing import Optional
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# subset["fail_flag"] = (
#         subset.get("facility_rating_status_y").fillna(1) <= 0
#     ).astype(int)

# MODEL_PATH = Path(__file__).parent / "model" / "xgboost_tuned_scaleweight.pkl"
MODEL_PATH='C:/Users/lkneh/HealthScore-Predictor/backend/model/xgboost_tuned_scaleweight.pkl'  # adjust as needed
raw_obj = joblib.load(MODEL_PATH)
_customer_join_cache = None
_neighborhood_feature_df = None
_google_clean_cache = None

NEIGHBORHOOD_SOURCE_PATH = "C:/Users/lkneh/HealthScore-Predictor/data/clean/Visualization_HealthInspections.csv"
GOOGLE_CLEAN_PATH = "C:/Users/lkneh/HealthScore-Predictor/data/clean/google_cleaned.csv"
TOP_SCORE_THRESHOLD = 3000
MIN_INSPECTION_DATE = pd.Timestamp("2023-01-01")


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

#input features used by the model
MODEL_FEATURES = [
    "facility_rating_status",
    "violation_count",
    "has_violation_count",
    "prev_rating_majority_3",
    "days_since_last_inspection",
    "avg_violation_count_last_3",
    "is_first_inspection",
    "analysis_neighborhood_Bernal Heights",
    "analysis_neighborhood_Castro/Upper Market",
    "analysis_neighborhood_Chinatown",
    "analysis_neighborhood_Excelsior",
    "analysis_neighborhood_Financial District/South Beach",
    "analysis_neighborhood_Glen Park",
    "analysis_neighborhood_Golden Gate Park",
    "analysis_neighborhood_Haight Ashbury",
    "analysis_neighborhood_Hayes Valley",
    "analysis_neighborhood_Inner Richmond",
    "analysis_neighborhood_Inner Sunset",
    "analysis_neighborhood_Japantown",
    "analysis_neighborhood_Lakeshore",
    "analysis_neighborhood_Lincoln Park",
    "analysis_neighborhood_Lone Mountain/USF",
    "analysis_neighborhood_Marina",
    "analysis_neighborhood_McLaren Park",
    "analysis_neighborhood_Mission",
    "analysis_neighborhood_Mission Bay",
    "analysis_neighborhood_Nob Hill",
    "analysis_neighborhood_Noe Valley",
    "analysis_neighborhood_North Beach",
    "analysis_neighborhood_Oceanview/Merced/Ingleside",
    "analysis_neighborhood_Outer Mission",
    "analysis_neighborhood_Outer Richmond",
    "analysis_neighborhood_Pacific Heights",
    "analysis_neighborhood_Portola",
    "analysis_neighborhood_Potrero Hill",
    "analysis_neighborhood_Presidio",
    "analysis_neighborhood_Presidio Heights",
    "analysis_neighborhood_Russian Hill",
    "analysis_neighborhood_Seacliff",
    "analysis_neighborhood_South of Market",
    "analysis_neighborhood_Sunset/Parkside",
    "analysis_neighborhood_Tenderloin",
    "analysis_neighborhood_Treasure Island",
    "analysis_neighborhood_Twin Peaks",
    "analysis_neighborhood_Visitacion Valley",
    "analysis_neighborhood_West of Twin Peaks",
    "analysis_neighborhood_Western Addition",
    "inspection_type_clean_complaint",
    "inspection_type_clean_complaint_reinspection",
    "inspection_type_clean_foodborne_illness",
    "inspection_type_clean_new_construction",
    "inspection_type_clean_new_ownership",
    "inspection_type_clean_new_ownership_followup",
    "inspection_type_clean_plan_check",
    "inspection_type_clean_plan_check_reinspection",
    "inspection_type_clean_reinspection",
    "inspection_type_clean_routine",
    "inspection_type_clean_site_visit",
    "inspection_type_clean_structural",
]

INSPECTION_OHE_FEATURES = [
    f for f in MODEL_FEATURES if f.startswith("inspection_type_clean_")
]

NEIGHBOURHOOD_OHE_FEATURES = [
    f for f in MODEL_FEATURES if f.startswith("analysis_neighborhood_")
]


# from frontend (compact payload)
class PredictionInput(BaseModel):
    facility_rating_status: float
    violation_count: float
    has_violation_count: int
    prev_rating_majority_3: float
    days_since_last_inspection: float
    avg_violation_count_last_3: float
    is_first_inspection: int
    analysis_neighborhood: str
    inspection_type: str


class PredictionOutput(BaseModel):
    # proba1: float
    # proba0: float
    risk_label: str
    risk_score: float
    outcome_label: int    # 0 or 1
    outcome_text: str 


class InspectorPredictionInput(BaseModel):
    business_id: Optional[int] = None
    business_name: Optional[str] = None
    inspection_type: str
    inspection_date: date


def make_features(data: PredictionInput) -> np.ndarray:
    # start with all zeros
    feats = {name: 0.0 for name in MODEL_FEATURES}

    # scalar features
    feats["facility_rating_status"] = float(data.facility_rating_status)
    feats["violation_count"] = float(data.violation_count)
    feats["has_violation_count"] = float(data.has_violation_count)
    feats["prev_rating_majority_3"] = float(data.prev_rating_majority_3)
    feats["days_since_last_inspection"] = float(data.days_since_last_inspection)
    feats["avg_violation_count_last_3"] = float(data.avg_violation_count_last_3)
    feats["is_first_inspection"] = float(data.is_first_inspection)

    # neighbourhood one-hot
    if data.analysis_neighborhood:
        neigh_col = data.analysis_neighborhood
        if neigh_col in NEIGHBOURHOOD_OHE_FEATURES:
            feats[neigh_col] = 1.0

    # inspection type one-hot
    col = f"inspection_type_clean_{data.inspection_type}"
    if col in INSPECTION_OHE_FEATURES:
        feats[col] = 1.0

    arr = np.array([feats[name] for name in MODEL_FEATURES], dtype=float)
    return arr.reshape(1, -1)


_insp_history_df = None
def load_inspection_history_df() -> pd.DataFrame:
    """Load and cache the encoded inspection history used for inspector predictions."""

    global _insp_history_df
    if _insp_history_df is not None:
        return _insp_history_df

    path = "C:/Users/lkneh/HealthScore-Predictor/data/clean/encoded/HealthInspectionsAll.csv"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("Error loading inspection history for /predict-inspector:", e)
        _insp_history_df = pd.DataFrame()
        return _insp_history_df

    _insp_history_df = df
    return _insp_history_df


def _safe_series_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    try:
        return float(val)
    except Exception:
        return default


def _clean_inspection_type_label(raw: str) -> str:
    """Map UI inspection type text to the cleaned labels used in MODEL_FEATURES.

    e.g. "Routine Inspection" -> "routine" so that
    col = "inspection_type_clean_routine" exists.
    """

    if raw is None:
        return "nan"
    s = str(raw).strip().lower()
    if s.endswith(" inspection"):
        s = s[: -len(" inspection")]
    return s or "nan"


def build_numeric_prediction_for_inspector(
    input_data: InspectorPredictionInput,
) -> PredictionInput:

    df = load_inspection_history_df()
    if df.empty:
        raise HTTPException(status_code=500, detail="Inspection history not available")

    biz_df = df.copy()

    # Prefer lookup by BusinessName_id if business_id is provided
    if input_data.business_id is not None:
        if "BusinessName_id" not in biz_df.columns:
            raise HTTPException(status_code=500, detail="Inspection history missing BusinessName_id column")
        biz_df = biz_df[biz_df["BusinessName_id"] == input_data.business_id]
    else:
        # Fallback: lookup by BusinessName string
        if "BusinessName" not in biz_df.columns:
            raise HTTPException(status_code=500, detail="Inspection history missing BusinessName column")

        if not input_data.business_name:
            raise HTTPException(status_code=400, detail="Business name or id is required")

        key = input_data.business_name.strip().upper()
        biz_df["_name_key"] = biz_df["BusinessName"].astype(str).str.strip().str.upper()
        biz_df = biz_df[biz_df["_name_key"] == key]

    if biz_df.empty:
        raise HTTPException(status_code=404, detail="No inspection history found for this business")

    # Build an inspection datetime for each historical record
    required_date_cols = {"insp_year", "insp_month", "insp_day"}
    if not required_date_cols.issubset(biz_df.columns):
        raise HTTPException(status_code=500, detail="Inspection history missing insp_year/month/day columns")

    biz_df = biz_df.copy()
    biz_df["_insp_datetime"] = pd.to_datetime(
        dict(
            year=biz_df["insp_year"],
            month=biz_df["insp_month"],
            day=biz_df["insp_day"],
        ),
        errors="coerce",
    )
    biz_df = biz_df.dropna(subset=["_insp_datetime"])
    if biz_df.empty:
        raise HTTPException(status_code=404, detail="No dated inspections found for this business")

    target_ts = pd.Timestamp(input_data.inspection_date)
    hist = biz_df[biz_df["_insp_datetime"] < target_ts].sort_values("_insp_datetime")

    if hist.empty:
        # If there is no prior inspection before the chosen date, fall back to the
        # latest known inspection for this business.
        last_row = biz_df.sort_values("_insp_datetime").iloc[-1]
    else:
        last_row = hist.iloc[-1]

    # Core numeric features inferred from the latest known inspection
    latitude = _safe_series_float(last_row, "latitude", 0.0)
    longitude = _safe_series_float(last_row, "longitude", 0.0)
    avg_viol_last_3 = _safe_series_float(last_row, "avg_violations_last_3", 0.0)
    fail_rate_last_3 = _safe_series_float(last_row, "fail_rate_last_3", 0.0)
    trend_last_3 = _safe_series_float(last_row, "trend_last_3", 0.0)

    # Days since last inspection: difference between requested date and last known inspection
    last_insp_ts = last_row["_insp_datetime"]
    days_since_last = max(0, int((target_ts - last_insp_ts).days))

    # Date‑based features for the *new* inspection
    insp_year = int(input_data.inspection_date.year)
    insp_month = int(input_data.inspection_date.month)
    insp_day = int(input_data.inspection_date.day)

    # Convert Python weekday (Mon=0..Sun=6) to 0=Sun..6=Sat to stay
    # consistent with how the UI labels days of week.
    dow_py = input_data.inspection_date.weekday()
    insp_dow = (dow_py + 1) % 7

    clean_type = _clean_inspection_type_label(input_data.inspection_type)

    return PredictionInput(
        latitude=latitude,
        longitude=longitude,
        avg_violations_last_3=avg_viol_last_3,
        fail_rate_last_3=fail_rate_last_3,
        days_since_last_inspection=days_since_last,
        trend_last_3=trend_last_3,
        insp_year=insp_year,
        insp_month=insp_month,
        insp_day=insp_day,
        insp_dow=insp_dow,
        # We currently do not recompute insp_days_since_ref; the
        # model has been operating with 0 here from the UI, so we
        # keep that behaviour.
        insp_days_since_ref=0,
        inspection_type=clean_type,
    )

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
    # print("Received? input:", input_data)
    X = make_features(input_data)
    print("Constructed features:", X)
    proba = float(model.predict_proba(X)[0,1])  # adjust if not binary
    print(proba)
    # p_fail = float(model.predict_proba(X)[0, 1])  # assuming classes_ == [0,1] and 1 == fail
    y_hat = int(model.predict(X)[0])
    # print(type(raw_obj), getattr(raw_obj, "keys", lambda: None)())

    # <0.4 = LOW, 0.4–0.7 = MEDIUM, >0.7 = HIGH.
    if proba > 0.7:
        label = "HIGH"
    elif proba >= 0.4:
        label = "MEDIUM"
    else:
        label = "LOW"
    outcome_text = "FAIL" if y_hat == 1 else "PASS"
    return PredictionOutput(risk_label=label, risk_score=proba,outcome_label=y_hat,outcome_text=outcome_text,)


@app.post("/predict-inspector", response_model=PredictionOutput)
def predict_inspector(input_data: InspectorPredictionInput):
    """Inspector‑focused prediction.

    Frontend sends only business name, inspection type, and target date.
    Backend reconstructs all engineered features from historical data.
    """

    numeric_input = build_numeric_prediction_for_inspector(input_data)
    X = make_features(numeric_input)
    proba = float(model.predict_proba(X)[0, 1])
    y_hat = int(model.predict(X)[0])

    if proba > 0.7:
        label = "HIGH"
    elif proba >= 0.4:
        label = "MEDIUM"
    else:
        label = "LOW"

    outcome_text = "FAIL" if y_hat == 1 else "PASS"
    return PredictionOutput(
        risk_label=label,
        risk_score=proba,
        outcome_label=y_hat,
        outcome_text=outcome_text,
    )
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




def load_customer_join_df():
    global _customer_join_cache
    if _customer_join_cache is not None:
        return _customer_join_cache

    base_df = _load_neighborhood_feature_df() #loades vis_data with cleanes analysis and name field
    if base_df is None or base_df.empty:
        _customer_join_cache = pd.DataFrame()
        return _customer_join_cache

    df = base_df.copy()
    df["name_key"] = df["name"].apply(_normalize_name)
    df = df[df["name_key"] != ""]
    # df.to_csv("customer_join_base_debug.csv", index=False)
    df["violation_count"] = pd.to_numeric(df.get("violation_count"), errors="coerce")
    df["facility_rating_status"] = pd.to_numeric(df.get("facility_rating_status"), errors="coerce")
    df["inspection_dt"] = pd.to_datetime(df.get("inspection_date"), errors="coerce")
    ratings = df["facility_rating_status"].fillna(1)
    df["fail_flag"] = np.where(ratings >= 2, 1, 0)
    # df.to_csv("customer_join_base_debug_after_failflage.csv", index=False)
    dedup_subset = ["name_key", "inspection_dt", "address"]
    df = df.sort_values("analysis_neighborhood")
    df = df.drop_duplicates(subset=dedup_subset, keep="first")
    # df.to_csv("customer_join_dedup_debug.csv", index=False)
    print(df[df["name_key"] == "THE MELT"].shape[0])
    grouped = (
        df.groupby(["name_key"])
        .agg(
            total_inspections=("name", "size"),
            total_violations=("violation_count", "sum"),
            avg_violation_count=("violation_count", "mean"),
            fails=("fail_flag", "sum"),
        )
        .reset_index()
    )
    # print("Grouped customer join df:", grouped.head())

    df["_inspection_order"] = df["inspection_dt"].fillna(pd.Timestamp("1900-01-01"))
    latest_idx = df.groupby("name_key")["_inspection_order"].idxmax()
    latest_details = df.loc[latest_idx, [
            "name_key",
            "name",
            "address",
            "latitude",
            "longitude",
            "inspection_dt",
            "facility_rating_status",
        ]].rename(columns={
            "name": "BusinessName",
            "inspection_dt": "last_inspection_date",
        })

    merged = grouped.merge(latest_details, on="name_key", how="left")
    merged["total_violations"] = merged["total_violations"].fillna(0)
    merged["avg_violation_count"] = merged["avg_violation_count"].fillna(0)
    merged["fail_rate"] = merged.apply(
        lambda r: float(r["fails"]) / float(r["total_inspections"]) if r["total_inspections"] else 0.0,
        axis=1,
    )
    # merged.to_csv("customer_join_merged_debug.csv", index=False)
    google_df = load_google_clean_df()
    if not google_df.empty:
        merged = merged.merge(
            google_df[
                [
                    "name_key",
                    "google_rating",
                    "google_reviews",
                    "google_address",
                    "google_lat",
                    "google_lon",
                ]
            ],
            on="name_key",
            how="left",
        )

        merged["address"] = merged["google_address"].combine_first(merged["address"])
        merged["latitude"] = merged["google_lat"].combine_first(merged["latitude"])
        merged["longitude"] = merged["google_lon"].combine_first(merged["longitude"])

    merged.drop(columns=["_inspection_order"], errors="ignore", inplace=True)

    _customer_join_cache = merged
    try:
        _customer_join_cache.to_csv("customer_join_debug.csv", index=False)
    except Exception:
        pass

    return _customer_join_cache


def _normalize_name(value) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


FACILITY_STATUS_LABELS = {
    0: "Pass",
    1: "Conditional Pass",
    2: "Fail",
}


def describe_facility_status(value) -> str:
    try:
        val = int(float(value))
    except (TypeError, ValueError):
        return "Unknown"
    return FACILITY_STATUS_LABELS.get(val, "Unknown")


#--------------------------for the cutomer niegbour endpoint------------------------
##load google clean dataframe
def load_google_clean_df() -> pd.DataFrame:
    global _google_clean_cache
    if _google_clean_cache is not None:
        return _google_clean_cache

    try:
        df = pd.read_csv(GOOGLE_CLEAN_PATH)
        df = df.copy()
        df["name_key"] = df["name"].apply(_normalize_name)
        df.rename(
            columns={
                "lat": "google_lat",
                "lng": "google_lon",
                "rating": "google_rating",
                "user_ratings_total": "google_reviews",
                "address": "google_address",
            },
            inplace=True,
        )
    except Exception as exc:
        print("Error loading google_cleaned.csv", exc)
        df = pd.DataFrame(columns=["name_key"])

    _google_clean_cache = df
    return _google_clean_cache

##for loading the neigbhorhood data from vis_hispections.csv
def _load_neighborhood_feature_df() -> pd.DataFrame:
    global _neighborhood_feature_df
    if _neighborhood_feature_df is not None:
        return _neighborhood_feature_df

    try:
        df = pd.read_csv(NEIGHBORHOOD_SOURCE_PATH)
        df = df.copy()
        df["neighborhood_key"] = (
            df["analysis_neighborhood"].astype(str).str.strip().str.lower()
        )
        df["name_key"] = df["name"].apply(_normalize_name)
    except Exception as exc:
        print("Error loading Visualization_HealthInspections.csv", exc)
        df = pd.DataFrame()

    _neighborhood_feature_df = df
    return _neighborhood_feature_df
#--------------------------------------------------

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
    base_df = _load_neighborhood_feature_df()
    if base_df is None or base_df.empty:
        summary = {
            "total_inspections": 0,
            "total_fails": 0,
            "overall_fail_rate": None,
            "risk_band_counts": [],
            "business_risk_bands": [],
            "high_risk_restaurants": None,
            "low_risk_restaurants": None,
        }
        return {
            "summary": summary,
            "inspection_trends": [],
            "inspection_type_stats": [],
            "inspection_delay_stats": [],
            "inspection_dow_stats": [],
            "dow_month_heatmap": [],
            "inspector_comparison": [],
            "neighbourhood_analysis": [],
            "inspection_year_pie": [],
            "high_risk_records": [],
            "violation_severity_pie": [],
        }

    # --- schema + preview for debugging/QA ---
    schema = []
    for col in base_df.columns:
        col_series = base_df[col]
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

    preview_raw = base_df.head(5).to_dict(orient="records")
    preview_rows = [
        {k: to_native(v) for k, v in row.items()}
        for row in preview_raw
    ]

    df = base_df.copy()
    df["name_key"] = df["name"].apply(_normalize_name)
    df = df[df["name_key"] != ""].copy()
    df["inspection_dt"] = pd.to_datetime(df.get("inspection_date"), errors="coerce")
    df = df.dropna(subset=["inspection_dt"]).copy()

    dedup_subset = ["name_key", "inspection_dt", "address"]
    df = df.sort_values("analysis_neighborhood")
    df = df.drop_duplicates(subset=dedup_subset, keep="first")

    numeric_cols = [
        "violation_count",
        "facility_rating_status",
        "has_violation_count",
        "prev_rating_majority_3",
        "days_since_last_inspection",
        "avg_violation_count_last_3",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "facility_rating_status" in df.columns:
        df["facility_rating_status"] = df["facility_rating_status"].fillna(1)
    else:
        df["facility_rating_status"] = 1

    df["facility_rating_status"] = pd.to_numeric(
        df["facility_rating_status"], errors="coerce"
    ).fillna(1)
    df["fail_flag"] = np.where(df["facility_rating_status"] >= 2, 1, 0)
    df["fail_status_exact"] = np.where(df["facility_rating_status"] == 2, 1, 0)

    df["insp_year"] = df["inspection_dt"].dt.year.astype("Int64")
    df["insp_month"] = df["inspection_dt"].dt.month.astype("Int64")
    df["insp_day"] = df["inspection_dt"].dt.day.astype("Int64")
    df["insp_dow"] = ((df["inspection_dt"].dt.dayofweek + 1) % 7).astype("Int64")

    df["BusinessName"] = df["name"].astype(str).str.strip()
    df = df[df["BusinessName"] != ""].copy()

    total_inspections = int(len(df))
    total_fails = int(df["fail_flag"].sum()) if total_inspections else 0
    overall_fail_rate = (
        float(total_fails / total_inspections) if total_inspections else None
    )

    risk_band_counts = []
    if total_inspections:
        band_counts = (
            df.groupby("facility_rating_status")["BusinessName"]
            .size()
            .reset_index(name="count")
            .sort_values("facility_rating_status")
        )
        risk_band_counts = [
            {
                "band": describe_facility_status(r["facility_rating_status"]),
                "rating_code": safe_float(r["facility_rating_status"]),
                "count": int(r["count"]),
            }
            for _, r in band_counts.iterrows()
        ]

    high_restaurants_count = None
    low_restaurants_count = None
    business_risk_bands = []
    biz_stats = None
    if total_inspections:
        biz_stats = (
            df.groupby("BusinessName")
            .agg(
                total_inspections=("fail_flag", "size"),
                fails=("fail_flag", "sum"),
                avg_violations=("violation_count", "mean"),
            )
            .reset_index()
        )
        high_restaurants_count = int((biz_stats["fails"] >= 1).sum())
        low_restaurants_count = int((biz_stats["fails"] == 0).sum())
        biz_stats["fail_rate"] = biz_stats["fails"] / biz_stats["total_inspections"]

        def _band(row):
            if row["fails"] == 0:
                return "Low (0 fails)"
            if row["fail_rate"] >= 0.5:
                return "High (>=50% fails)"
            return "Medium"

        biz_stats["risk_band"] = biz_stats.apply(_band, axis=1)
        band_counts = (
            biz_stats.groupby("risk_band")["BusinessName"]
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

    inspection_trends = []
    if total_inspections and df["insp_year"].notna().any():
        trend_counts = (
            df.groupby(["insp_year", "facility_rating_status"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        for year, row in trend_counts.iterrows():
            inspection_trends.append(
                {
                    "year": int(to_native(year)),
                    "pass_count": int(to_native(row.get(0, 0))),
                    "conditional_count": int(to_native(row.get(1, 0))),
                    "fail_count": int(to_native(row.get(2, 0))),
                }
            )

    inspection_type_stats = []
    if "inspection_type_clean" in df.columns and total_inspections:
        type_g = (
            df.groupby("inspection_type_clean")["fail_flag"]
            .agg(total="size", fails="sum")
            .reset_index()
        )
        type_g["fail_rate"] = type_g["fails"] / type_g["total"]
        type_g = type_g.sort_values("total", ascending=False).head(10)
        inspection_type_stats = [
            {
                "inspection_type": str(to_native(r.inspection_type_clean)).replace("_", " ").title(),
                "total": int(to_native(r.total)),
                "fails": int(to_native(r.fails)),
                "fail_rate": float(to_native(r.fail_rate)),
            }
            for _, r in type_g.iterrows()
        ]

    inspection_dow_stats = []
    if total_inspections and df["insp_dow"].notna().any():
        dow_df = df[["insp_dow", "fail_flag"]].dropna()
        dow_df["insp_dow"] = dow_df["insp_dow"].astype(int)
        dow_g = (
            dow_df.groupby("insp_dow")["fail_flag"]
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
            for _, r in dow_g.sort_values("insp_dow").iterrows()
        ]

    inspection_delay_stats = []
    if total_inspections and "days_since_last_inspection" in df.columns:
        delay_df = df[["days_since_last_inspection", "fail_flag"]].copy()
        delay_df["days_since_last_inspection"] = pd.to_numeric(
            delay_df["days_since_last_inspection"], errors="coerce"
        )
        delay_df = delay_df.dropna(subset=["days_since_last_inspection"])
        if not delay_df.empty:
            bins = [-1, 0, 30, 90, 180, 365, float("inf")]
            labels = ["0", "1–30", "31–90", "91–180", "181–365", "366+"]
            delay_df["bucket"] = pd.cut(delay_df["days_since_last_inspection"], bins=bins, labels=labels)
            d_g = (
                delay_df.groupby("bucket")["fail_flag"]
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

    neighborhood_fail_stats = []
    if total_inspections and "analysis_neighborhood" in df.columns:
        n_df = df[["analysis_neighborhood", "fail_flag", "fail_status_exact"]].copy()
        n_df["analysis_neighborhood"] = n_df["analysis_neighborhood"].astype(str).str.strip()
        n_df = n_df[n_df["analysis_neighborhood"] != ""]
        if not n_df.empty:
            n_grouped = (
                n_df.groupby("analysis_neighborhood")
                .agg(
                    total=("fail_flag", "size"),
                    fail_records=("fail_status_exact", "sum"),
                )
                .reset_index()
            )
            n_grouped = n_grouped[n_grouped["fail_records"] > 0]
            n_grouped = n_grouped.sort_values("fail_records", ascending=False)
            neighborhood_fail_stats = [
                {
                    "neighborhood": to_native(r.analysis_neighborhood),
                    "total": int(to_native(r.total)),
                    "fails": int(to_native(r.fail_records)),
                    "fail_rate": float(r.fail_records / r.total) if r.total else 0.0,
                }
                for _, r in n_grouped.iterrows()
            ]

    dow_month_heatmap = []
    if total_inspections:
        cal_df = df[["insp_month", "insp_dow", "fail_flag", "violation_count"]].copy()
        cal_df = cal_df.dropna(subset=["insp_month", "insp_dow"])
        if not cal_df.empty:
            cal_df["insp_month"] = cal_df["insp_month"].astype(int)
            cal_df["insp_dow"] = cal_df["insp_dow"].astype(int)
            agg_dict = {
                "fail_flag": ["size", "mean"],
                "violation_count": "mean",
            }
            cal_g = cal_df.groupby(["insp_month", "insp_dow"]).agg(agg_dict).reset_index()
            cal_g.columns = ["insp_month", "insp_dow", "total", "fail_rate", "avg_violations"]
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
                dow_month_heatmap.append(
                    {
                        "month_index": int(to_native(r.insp_month)),
                        "month": month_labels.get(int(to_native(r.insp_month)), str(to_native(r.insp_month))),
                        "dow_index": int(to_native(r.insp_dow)),
                        "dow": dow_labels.get(int(to_native(r.insp_dow)), str(to_native(r.insp_dow))),
                        "total": int(to_native(r.total)),
                        "fail_rate": float(to_native(r.fail_rate)),
                        "avg_violations": float(to_native(r.avg_violations)),
                    }
                )

    inspector_comparison = []  # BusinessName_id column not present in visualization dataset

    neighbourhood_analysis = []
    if total_inspections and {"latitude", "longitude"}.issubset(df.columns):
        loc = df[["latitude", "longitude", "fail_flag"]].dropna()
        if not loc.empty:
            loc["lat_bin"] = loc["latitude"].round(3)
            loc["lon_bin"] = loc["longitude"].round(3)
            neigh = (
                loc.groupby(["lat_bin", "lon_bin"]).agg(total=("fail_flag", "size"), fails=("fail_flag", "sum"))
                .reset_index()
            )
            neigh["fail_rate"] = neigh["fails"] / neigh["total"]
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
    if total_inspections:
        year_counts = df["insp_year"].dropna().value_counts().sort_index()
        inspection_year_pie = [
            {
                "label": str(int(to_native(year))),
                "value": int(to_native(count)),
            }
            for year, count in year_counts.items()
        ]

    high_risk_records = []
    if biz_stats is not None and not biz_stats.empty:
        latest_idx = df.groupby("BusinessName")["inspection_dt"].idxmax()
        latest_status = df.loc[
            latest_idx,
            ["BusinessName", "inspection_dt", "facility_rating_status"],
        ]
        latest_status = latest_status.rename(columns={"inspection_dt": "last_inspection"})
        latest_status["insp_year"] = latest_status["last_inspection"].dt.year.astype("Int64")

        high_risk = biz_stats.merge(latest_status, on="BusinessName", how="left")
        high_risk = high_risk[high_risk["fails"] >= 1]
        if not high_risk.empty:
            high_risk = high_risk.sort_values(
                ["fails", "fail_rate", "avg_violations"], ascending=False
            ).head(30)
            high_risk_records = [
                {
                    "BusinessName": to_native(r.BusinessName),
                    "year": int(to_native(r.insp_year)) if pd.notna(r.insp_year) else None,
                    "total_inspections": int(to_native(r.total_inspections)),
                    "fails": int(to_native(r.fails)),
                    "fail_rate": float(to_native(r.fail_rate)),
                    "avg_violations": float(to_native(r.avg_violations)) if pd.notna(r.avg_violations) else None,
                    "facility_rating_status": safe_float(r.facility_rating_status),
                    "status_label": describe_facility_status(r.facility_rating_status),
                }
                for _, r in high_risk.iterrows()
            ]

    violation_severity_pie = []
    if "violation_count" in df.columns and total_inspections:
        severity = pd.cut(df["violation_count"], bins=[-1, 0, 2, 5, float("inf")], labels=["0", "1–2", "3–5", "6+"])
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
        "neighborhood_fail_stats": neighborhood_fail_stats,
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

    df = load_customer_join_df()
    if df is None or df.empty:
        return {"error": "Business insights not available"}

    key = _normalize_name(name)
    if not key:
        return {"error": "Business name is required"}

    exact = df[df["name_key"] == key]
    if not exact.empty:
        candidates = exact
    else:
        candidates = df[df["name_key"].str.contains(key, na=False)]

    if candidates.empty:
        return {"error": "No matching business found"}

    row = (
        candidates.sort_values(
            ["total_inspections", "last_inspection_date"],
            ascending=[False, False],
        ).iloc[0]
    )

    resp = {
        "business_name": to_native(row.get("BusinessName")) or name.strip(),
        "address": to_native(row.get("address")),
        "latitude": safe_float(row.get("latitude")),
        "longitude": safe_float(row.get("longitude")),
        "total_inspections": int(safe_float(row.get("total_inspections")) or 0),
        "total_violations": int(safe_float(row.get("total_violations")) or 0),
        "fails": int(safe_float(row.get("fails")) or 0),
        "fail_rate": safe_float(row.get("fail_rate")),
        "google_rating": safe_float(row.get("google_rating")),
        "avg_violation_count": safe_float(row.get("avg_violation_count")),
        "last_inspection_date": to_native(row.get("last_inspection_date")),
    }

    reviews = row.get("google_reviews")
    try:
        resp["google_reviews"] = int(reviews) if pd.notna(reviews) else None
    except Exception:
        resp["google_reviews"] = None

    return resp


@app.get("/customer-neighborhoods")
def customer_neighborhoods(name: Optional[str] = None):

    df = _load_neighborhood_feature_df()
    google_df = load_google_clean_df()

    if df is None or df.empty:
        return {"options": [], "message": "Neighborhood feature dataset not available"}

    options = (
        df["analysis_neighborhood"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique()
    )
    payload = {"options": sorted(options.tolist())}

    if not name:
        return payload

    key = name.strip().lower()
    subset = df[df["neighborhood_key"] == key].copy()
    if subset.empty:
        raise HTTPException(status_code=404, detail="Neighborhood not recognized or has no records")

    numeric_cols = [
        "violation_count",
        "avg_violation_count_last_3",
        "days_since_last_inspection",
        "is_first_inspection",
        "prev_rating_majority_3",
        "facility_rating_status",
        "has_violation_count",
    ]
    for col in numeric_cols:
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")

    subset["inspection_dt"] = pd.to_datetime(subset.get("inspection_date"), errors="coerce")
    rating_series = subset.get("facility_rating_status")
    if rating_series is not None:
        subset["fail_flag"] = (rating_series.fillna(1) >= 2).astype(int)
    else:
        subset["fail_flag"] = 0

    def _mean_of(col_name: str):
        if col_name not in subset.columns:
            return None
        series = subset[col_name].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return None
        return safe_float(float(series.mean()))

    total_inspections = int(len(subset))

    mix = []
    if "inspection_type_clean" in subset.columns:
        mix_counts = subset["inspection_type_clean"].dropna().value_counts().head(10)
        total_mix = float(mix_counts.sum()) if len(mix_counts) else 0.0
        for label, count in mix_counts.items():
            entry = {
                "inspection_type": str(label).replace("_", " ").title(),
                "count": int(count),
            }
            if total_mix > 0:
                entry["share"] = safe_float(count / total_mix)
            mix.append(entry)

    summary = {
        "neighborhood": name.strip(),
        "total_inspections": total_inspections,
        "avg_violation_count": _mean_of("violation_count"),
        "avg_violation_count_last_3": _mean_of("avg_violation_count_last_3"),
        "avg_days_since_last_inspection": _mean_of("days_since_last_inspection"),
        "first_inspection_rate": _mean_of("is_first_inspection"),
        "avg_prev_rating_majority_3": _mean_of("prev_rating_majority_3"),
        "avg_facility_rating_status": _mean_of("facility_rating_status"),
        "inspection_mix": mix,
    }

    biz = subset.copy()
    biz["name_key"] = biz["name"].apply(_normalize_name)

    dedup_subset = ["name_key", "inspection_dt", "address"]
    biz = biz.sort_values("analysis_neighborhood")
    biz = biz.drop_duplicates(subset=dedup_subset, keep="first")

    grouped = (
        biz.groupby("name_key")
        .agg(
            total_inspections=("name", "size"),
            total_violations=("violation_count", "sum"),
            avg_violation_count=("violation_count", "mean"),
            avg_days_since_last=("days_since_last_inspection", "mean"),
            first_inspection_rate=("is_first_inspection", "mean"),
            avg_prev_rating=("prev_rating_majority_3", "mean"),
            fails=("fail_flag", "sum"),
        )
        .reset_index()
    )
    
    grouped["fail_rate"] = grouped.apply(
        lambda r: float(r["fails"]) / float(r["total_inspections"]) if r["total_inspections"] else 0.0,
        axis=1,
    )

    latest_idx = biz.groupby("name_key")["inspection_dt"].idxmax()
    latest_details = (
        biz.loc[latest_idx, [
            "name_key",
            "inspection_dt",
            "facility_rating_status",
            "address",
            "name",
            "latitude",
            "longitude",
        ]]
        .rename(columns={"name": "display_name"})
    )
    business_df = grouped.merge(latest_details, on="name_key", how="left")

    if not google_df.empty:
        business_df = business_df.merge(
            google_df[
                [
                    "name_key",
                    "google_rating",
                    "google_reviews",
                    "google_address",
                    "google_lat",
                    "google_lon",
                ]
            ],
            on="name_key",
            how="left",
        )

    business_df["google_rating"] = pd.to_numeric(
        business_df.get("google_rating"), errors="coerce"
    )
    business_df["google_reviews"] = pd.to_numeric(
        business_df.get("google_reviews"), errors="coerce"
    )
    business_df["avg_violation_count"] = pd.to_numeric(
        business_df.get("avg_violation_count"), errors="coerce"
    )
    business_df["avg_days_since_last"] = pd.to_numeric(
        business_df.get("avg_days_since_last"), errors="coerce"
    )
    business_df["fail_rate"] = pd.to_numeric(
        business_df.get("fail_rate"), errors="coerce"
    )

    inspection_series = pd.to_datetime(
        business_df.get("inspection_dt"), errors="coerce", utc=True
    )
    now_utc = pd.Timestamp.now(tz="UTC")
    business_df["days_since_latest"] = (
        now_utc - inspection_series
    ).dt.days
    business_df["inspection_dt"] = inspection_series.dt.tz_convert("UTC").dt.tz_localize(None)

    recent_window_days = 365.0
    business_df["recent_factor"] = (
        recent_window_days
        - business_df["days_since_latest"].clip(lower=0).fillna(recent_window_days)
    )
    business_df["recent_factor"] = (
        business_df["recent_factor"].clip(lower=0) / recent_window_days
    )
    business_df["score"] = (
        business_df["google_rating"].fillna(0) * 1000
        + business_df["google_reviews"].fillna(0)
        - business_df["total_violations"].fillna(0) * 10
        - business_df["fail_rate"].fillna(0) * 1000
        # + business_df["recent_factor"].fillna(0) * 500
    )
    # business_df.to_csv("debug_neighborhood_business_scores.csv", index=False)
    eligible = business_df[
        # (business_df["score"] > 1000)
        # & (business_df["google_reviews"] > 1000)
        # & business_df["inspection_dt"].notna()
        # & (business_df["inspection_dt"] >= MIN_INSPECTION_DATE)
        (business_df["avg_violation_count"] == 0)
        & (business_df["facility_rating_status"] == 0)
    ]

    if eligible.empty:
        payload["summary"] = summary
        payload["top_restaurants"] = []
        payload["map_points"] = []
        payload["message"] = (
            "No high-scoring restaurants with recent inspections were found."
        )
        return payload

    business_df = eligible.sort_values("score", ascending=False).head(10)
    # business_df = business_df.sort_values("inspection_dt", ascending=False)
    

    top_restaurants = []
    map_points = []
    for _, row in business_df.iterrows():
        lat = row.get("google_lat")
        lon = row.get("google_lon")
        if pd.isna(lat) or pd.isna(lon):
            lat = row.get("latitude")
            lon = row.get("longitude")

        entry = {
            "name": to_native(row.get("display_name")) or to_native(row.get("name_key")),
            "address": to_native(row.get("google_address")) or to_native(row.get("address")),
            "rating": safe_float(row.get("google_rating")),
            "user_ratings_total": int(row.get("google_reviews")) if pd.notna(row.get("google_reviews")) else None,
            "total_inspections": int(row.get("total_inspections", 0) or 0),
            "total_violations": safe_float(row.get("total_violations")),
            "avg_violation_count": safe_float(row.get("avg_violation_count")),
            "last_inspection_date": to_native(row.get("inspection_dt")),
            "facility_rating_status": safe_float(row.get("facility_rating_status")),
            "days_since_last_inspection": (
                int(row.get("days_since_latest"))
                if pd.notna(row.get("days_since_latest"))
                else None
            ),
            "fail_rate": safe_float(row.get("fail_rate")),
            "lat": safe_float(lat),
            "lon": safe_float(lon),
        }
        top_restaurants.append(entry)

        if entry["lat"] is not None and entry["lon"] is not None:
            map_points.append({
                "name": entry["name"],
                "lat": entry["lat"],
                "lon": entry["lon"],
                "rating": entry["rating"],
            })

    payload["summary"] = summary
    payload["top_restaurants"] = top_restaurants
    payload["map_points"] = map_points
    payload.pop("message", None)
    return payload
