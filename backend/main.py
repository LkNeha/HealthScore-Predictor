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



# MODEL_PATH = Path(__file__).parent / "model" / "xgboost_tuned_scaleweight.pkl"
MODEL_PATH = 'C:/Users/lkneh/HealthScore-Predictor/backend/model/xgboost_tuned_scaleweight.pkl'  # adjust as needed
raw_obj = joblib.load(MODEL_PATH)
_customer_join_cache = None
_neighborhood_feature_df = None
_google_clean_cache = None
_rf_multiclass_model = None
_xgb_multiclass_model = None
_xgb_multiclass_scaler = None
_model_dataset_cache = None

NEIGHBORHOOD_SOURCE_PATH = "C:/Users/lkneh/HealthScore-Predictor/data/clean/Visualization_HealthInspections.csv"
GOOGLE_CLEAN_PATH = "C:/Users/lkneh/HealthScore-Predictor/data/clean/google_cleaned.csv"
MULTICLASS_DATASET_PATH = "C:/Users/lkneh/HealthScore-Predictor/data/clean/model_dataset.csv"
RF_MULTICLASS_MODEL_PATH = "C:/Users/lkneh/HealthScore-Predictor/notebooks/Model/random_forest_multiclass.pkl"
XGB_MULTICLASS_MODEL_PATH = "C:/Users/lkneh/HealthScore-Predictor/notebooks/Model/xgboost_multiclass_model_20260115_000350.pkl"
MULTICLASS_SCALER_PATH = "C:/Users/lkneh/HealthScore-Predictor/notebooks/Model/scaler_20260115_000350.pkl"
METRICS_PKL_PATH = "C:/Users/lkneh/HealthScore-Predictor/notebooks/Model/model_metrics_20260115_000350.pkl"
RF_METRICS_PKL_PATH = "C:/Users/lkneh/HealthScore-Predictor/notebooks/Model/rfmodel_metrics.pkl"
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
    # "facility_rating_status",
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


# from frontend 
class PredictionInput(BaseModel):
    violation_count: float
    has_violation_count: int
    prev_rating_majority_3: float
    days_since_last_inspection: float
    avg_violation_count_last_3: float
    is_first_inspection: int
    analysis_neighborhood: str
    inspection_type: str


class PredictRequest(BaseModel):
    business_name: str
    inspection_type: str
    inspection_date: date
    analysis_neighborhood: str
    violation_count: Optional[float] = None


class PredictionOutput(BaseModel):
    # proba1: float
    # proba0: float
    risk_label: str
    risk_score: float
    outcome_label: int    # 0 or 1
    outcome_text: str
    proba_pass: Optional[float] = None
    proba_conditional: Optional[float] = None
    proba_fail: Optional[float] = None
    predicted_class_label: Optional[str] = None


def make_features(data: PredictionInput) -> np.ndarray:
    # start with all zeros
    feats = {name: 0.0 for name in MODEL_FEATURES}

    # scalar features
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

    # Debug: show only the non-zero features going into the model
    try:
        non_zero = {k: v for k, v in feats.items() if v != 0.0}
        print("[DEBUG] /predict feature_vector_non_zero", non_zero)
    except Exception:
        pass

    return arr.reshape(1, -1)

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
    if raw is None:
        return "nan"
    s = str(raw).strip().lower()
    if s.endswith(" inspection"):
        s = s[: -len(" inspection")]
    return s or "nan"

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


def _load_multiclass_model_and_scaler():
    global _rf_multiclass_model, _xgb_multiclass_scaler

    if _rf_multiclass_model is None:
        obj = joblib.load(RF_MULTICLASS_MODEL_PATH)
        if isinstance(obj, dict) and "model" in obj:
            _rf_multiclass_model = obj["model"]
        else:
            _rf_multiclass_model = obj

    if _xgb_multiclass_scaler is None:
        _xgb_multiclass_scaler = joblib.load(MULTICLASS_SCALER_PATH)

    return _rf_multiclass_model, _xgb_multiclass_scaler


def _load_model_dataset_df() -> pd.DataFrame:

    global _model_dataset_cache

    if _model_dataset_cache is not None:
        return _model_dataset_cache

    try:
        df = pd.read_csv(MULTICLASS_DATASET_PATH)
    except Exception as exc:
        print("Error loading model_dataset.csv", exc)
        df = pd.DataFrame()

    _model_dataset_cache = df
    return _model_dataset_cache


def _build_multiclass_features_from_visualization(payload: PredictRequest) -> np.ndarray:
    df = _load_neighborhood_feature_df()
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="Visualization dataset not available")

    if "inspection_date" not in df.columns:
        raise HTTPException(status_code=500, detail="inspection_date column missing in visualization dataset")

    name_key = _normalize_name(payload.business_name)

    # analysis_neighborhood from the frontend is a one-hot feature name like
    # "analysis_neighborhood_Mission"; convert it back to the raw label
    raw_neigh_from_input = str(payload.analysis_neighborhood or "").strip()
    if raw_neigh_from_input.startswith("analysis_neighborhood_"):
        raw_neigh_label = raw_neigh_from_input[len("analysis_neighborhood_") :]
    else:
        raw_neigh_label = raw_neigh_from_input
    neigh_key = raw_neigh_label.strip().lower()

    subset = df[(df["name_key"] == name_key) & (df["neighborhood_key"] == neigh_key)]
    if subset.empty:
        subset = df[df["name_key"] == name_key]

    if subset.empty:
        raise HTTPException(status_code=404, detail="Business not found in visualization dataset")

    # Debug: show how many matches we found
    try:
        print(
            "[DEBUG] /predict lookup",
            {
                "business_name": payload.business_name,
                "analysis_neighborhood_input": payload.analysis_neighborhood,
                "name_key": name_key,
                "neigh_key": neigh_key,
                "matched_rows": int(len(subset)),
            },
        )
    except Exception:
        pass

    subset = subset.copy()
    subset["inspection_date_parsed"] = pd.to_datetime(subset["inspection_date"], errors="coerce")
    subset = subset.dropna(subset=["inspection_date_parsed"])
    if subset.empty:
        raise HTTPException(status_code=500, detail="No valid inspection dates for this business")

    # Take the latest inspection by date as the reference record
    ref_row = subset.sort_values("inspection_date_parsed").iloc[-1]
    last_insp_date = ref_row["inspection_date_parsed"].date()
    target_date = payload.inspection_date

    days_since_last = max((target_date - last_insp_date).days, 0)

    # Scalar features for the current inspection violations.
    # By default, if the inspector does not provide a violation_count,
    # we assume 0 violations for this inspection, but we now
    # always set has_violation_count = 1 as requested.
    violation_count = 0.0
    has_violation_count = 1

    # If the inspector provides a current violation_count in the request,
    # use that value, but keep has_violation_count hard-coded to 1.
    if payload.violation_count is not None:
        try:
            violation_count = float(payload.violation_count)
        except Exception:
            # If parsing fails, keep the default 0.0
            violation_count = 0.0
    prev_rating_majority_3 = _safe_series_float(ref_row, "prev_rating_majority_3", default=0.0)
    avg_viol_last_3 = _safe_series_float(ref_row, "avg_violation_count_last_3", default=0.0)
    is_first_inspection = int(_safe_series_float(ref_row, "is_first_inspection", default=0.0))

    # One-hot neighborhood column name should align with frontend-selected analysis_neighborhood
    raw_neigh = str(ref_row.get("analysis_neighborhood", "")).strip()
    neigh_feature = payload.analysis_neighborhood

    # Clean inspection type from UI into the short form used in MODEL_FEATURES
    clean_type = _clean_inspection_type_label(payload.inspection_type)

    numeric_input = PredictionInput(
        violation_count=violation_count,
        has_violation_count=has_violation_count,
        prev_rating_majority_3=prev_rating_majority_3,
        days_since_last_inspection=float(days_since_last),
        avg_violation_count_last_3=avg_viol_last_3,
        is_first_inspection=is_first_inspection,
        analysis_neighborhood=neigh_feature,
        inspection_type=clean_type,
    )

    # Debug: show the scalar inputs and categorical labels used to build features
    try:
        print(
            "[DEBUG] /predict numeric_input",
            {
                **numeric_input.dict(),
                "raw_neighborhood_csv": raw_neigh,
                "neigh_feature_used": neigh_feature,
            },
        )
    except Exception:
        pass

    return make_features(numeric_input)


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictRequest):
    rf_model, scaler_mc = _load_multiclass_model_and_scaler()

    X_raw = _build_multiclass_features_from_visualization(payload)
    X_scaled = scaler_mc.transform(X_raw)

    proba_all = rf_model.predict_proba(X_scaled)[0]
    classes = getattr(rf_model, "classes_", np.array([0, 1, 2]))

    # Probability of FAIL (class 2) for risk banding
    fail_index = None
    try:
        idx = np.where(classes == 2)[0]
        if len(idx) > 0:
            fail_index = int(idx[0])
    except Exception:
        fail_index = None
    if fail_index is None:
        fail_index = proba_all.shape[0] - 1

    # Map full probability vector into class-wise values
    # Assume classes are 0, 1, 2 = Pass, Conditional Pass, Fail
    def _get_proba_for_class(cls_id: int) -> float:
        try:
            idx_arr = np.where(classes == cls_id)[0]
            if len(idx_arr) == 0:
                return 0.0
            return float(proba_all[int(idx_arr[0])])
        except Exception:
            return 0.0

    p_pass = _get_proba_for_class(0)
    p_conditional = _get_proba_for_class(1)
    p_fail = _get_proba_for_class(2)

    y_hat = int(rf_model.predict(X_scaled)[0])

    # Map probability to LOW / MEDIUM / HIGH as before
    if p_fail > 0.7:
        label = "HIGH"
    elif p_fail >= 0.4:
        label = "MEDIUM"
    else:
        label = "LOW"

    # Binary outcome for UI: FAIL if predicted class == 2, else PASS
    outcome_binary = 1 if y_hat == 2 else 0
    if y_hat == 0:
        predicted_label = "Pass"
    elif y_hat == 1:
        predicted_label = "Conditional Pass"
    elif y_hat == 2:
        predicted_label = "Fail"
    else:
        predicted_label = "Unknown"

    outcome_text = "FAIL" if outcome_binary == 1 else "PASS"

    return PredictionOutput(
        risk_label=label,
        risk_score=p_fail,
        outcome_label=outcome_binary,
        outcome_text=outcome_text,
        proba_pass=p_pass,
        proba_conditional=p_conditional,
        proba_fail=p_fail,
        predicted_class_label=predicted_label,
    )


@app.get("/debug-fail-samples")
def debug_fail_samples(limit: int = 5):
    xgb_model, scaler_mc = _load_multiclass_model_and_scaler()
    df = _load_model_dataset_df()

    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="model_dataset.csv not available")

    if "facility_rating_status" not in df.columns:
        raise HTTPException(status_code=500, detail="facility_rating_status column missing in model_dataset.csv")

    # True FAIL rows from the training dataset
    fail_df = df[df["facility_rating_status"] == 2].copy()
    if fail_df.empty:
        raise HTTPException(status_code=500, detail="No FAIL rows (status=2) found in model_dataset.csv")

    fail_df = fail_df.reset_index().rename(columns={"index": "row_index"})
    fail_df = fail_df.head(max(1, int(limit)))

    # MODEL_FEATURES are exactly the columns used during training
    missing_feats = [c for c in MODEL_FEATURES if c not in fail_df.columns]
    if missing_feats:
        raise HTTPException(
            status_code=500,
            detail=f"model_dataset.csv missing expected feature columns: {missing_feats}",
        )

    X = fail_df[MODEL_FEATURES].values.astype(float)
    y_true = fail_df["facility_rating_status"].astype(int).values

    X_scaled = scaler_mc.transform(X)
    proba_all = xgb_model.predict_proba(X_scaled)
    y_pred = xgb_model.predict(X_scaled).astype(int)

    classes = getattr(xgb_model, "classes_", np.array([0, 1, 2]))

    def _proba_for(sample_idx: int, cls_id: int) -> float:
        try:
            idx_arr = np.where(classes == cls_id)[0]
            if len(idx_arr) == 0:
                return 0.0
            return float(proba_all[sample_idx, int(idx_arr[0])])
        except Exception:
            return 0.0

    results = []
    for i in range(len(fail_df)):
        row = fail_df.iloc[i]
        results.append(
            {
                "row_index": int(row["row_index"]),
                "true_label": int(y_true[i]),  # should be 2 (FAIL)
                "proba_pass": _proba_for(i, 0),
                "proba_conditional": _proba_for(i, 1),
                "proba_fail": _proba_for(i, 2),
                "predicted_label": int(y_pred[i]),
            }
        )

    return {"samples": results, "classes": classes.tolist()}


@app.get("/debug-fail-input-examples")
def debug_fail_input_examples(
    limit: int = 5,
    threshold: float = 0.7,
    violation_count: Optional[float] = None,
    inspection_date: Optional[date] = None,
):
    xgb_model, scaler_mc = _load_multiclass_model_and_scaler()
    df = _load_neighborhood_feature_df()

    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="Visualization dataset not available")

    if "inspection_date" not in df.columns:
        raise HTTPException(status_code=500, detail="inspection_date column missing in visualization dataset")

    tmp = df.copy()
    tmp["inspection_date_parsed"] = pd.to_datetime(tmp["inspection_date"], errors="coerce")
    tmp = tmp.dropna(subset=["inspection_date_parsed"])

    if "name_key" not in tmp.columns:
        tmp["name_key"] = tmp["name"].apply(_normalize_name)
    if "neighborhood_key" not in tmp.columns:
        tmp["neighborhood_key"] = (
            tmp["analysis_neighborhood"].astype(str).str.strip().str.lower()
        )

    # Latest inspection per (business, neighborhood)
    latest_idx = tmp.groupby(["name_key", "neighborhood_key"]) ["inspection_date_parsed"].idxmax()
    latest = tmp.loc[latest_idx].reset_index(drop=True)

    classes = getattr(xgb_model, "classes_", np.array([0, 1, 2]))

    def _proba_for(proba_vec: np.ndarray, cls_id: int) -> float:
        try:
            idx_arr = np.where(classes == cls_id)[0]
            if len(idx_arr) == 0:
                return 0.0
            return float(proba_vec[int(idx_arr[0])])
        except Exception:
            return 0.0

    examples = []

    for _, row in latest.iterrows():
        business_name = str(row.get("name", "")).strip()
        neigh_label = str(row.get("analysis_neighborhood", "")).strip()
        insp_dt_hist = row["inspection_date_parsed"].date()
        insp_type_clean = str(row.get("inspection_type_clean", "routine") or "routine").strip().lower()

        if not business_name or not neigh_label:
            continue

        analysis_neighborhood_feature = f"analysis_neighborhood_{neigh_label}"

        # Target inspection date for the model: either the caller-provided
        # inspection_date or the historical inspection date from the dataset.
        target_dt = inspection_date or insp_dt_hist

        payload = PredictRequest(
            business_name=business_name,
            inspection_type=insp_type_clean,
            inspection_date=target_dt,
            analysis_neighborhood=analysis_neighborhood_feature,
            violation_count=violation_count,
        )

        try:
            X_raw = _build_multiclass_features_from_visualization(payload)
            X_scaled = scaler_mc.transform(X_raw)
            proba_vec = xgb_model.predict_proba(X_scaled)[0]
            y_hat = int(xgb_model.predict(X_scaled)[0])
        except Exception:
            continue

        p_fail = _proba_for(proba_vec, 2)
        p_pass = _proba_for(proba_vec, 0)
        p_cond = _proba_for(proba_vec, 1)

        if y_hat == 2 and p_fail >= float(threshold):
            examples.append(
                {
                    "business_name": business_name,
                    "analysis_neighborhood_input": analysis_neighborhood_feature,
                    "analysis_neighborhood_label": neigh_label,
                    "inspection_type_input": insp_type_clean,
                    "inspection_date_input": target_dt.isoformat(),
                    "violation_count_input": violation_count,
                    "proba_pass": p_pass,
                    "proba_conditional": p_cond,
                    "proba_fail": p_fail,
                    "predicted_label": y_hat,
                }
            )
            if len(examples) >= max(1, int(limit)):
                break

    return {
        "threshold": float(threshold),
        "violation_count": violation_count,
        "inspection_date": inspection_date.isoformat() if inspection_date else None,
        "examples": examples,
    }


@app.get("/debug-conditional-input-examples")
def debug_conditional_input_examples(
    limit: int = 5,
    threshold: float = 0.7,
    violation_count: Optional[float] = None,
    inspection_date: Optional[date] = None,
):
    xgb_model, scaler_mc = _load_multiclass_model_and_scaler()
    df = _load_neighborhood_feature_df()

    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="Visualization dataset not available")

    if "inspection_date" not in df.columns:
        raise HTTPException(status_code=500, detail="inspection_date column missing in visualization dataset")

    tmp = df.copy()
    tmp["inspection_date_parsed"] = pd.to_datetime(tmp["inspection_date"], errors="coerce")
    tmp = tmp.dropna(subset=["inspection_date_parsed"])

    if "name_key" not in tmp.columns:
        tmp["name_key"] = tmp["name"].apply(_normalize_name)
    if "neighborhood_key" not in tmp.columns:
        tmp["neighborhood_key"] = (
            tmp["analysis_neighborhood"].astype(str).str.strip().str.lower()
        )

    latest_idx = tmp.groupby(["name_key", "neighborhood_key"]) ["inspection_date_parsed"].idxmax()
    latest = tmp.loc[latest_idx].reset_index(drop=True)

    classes = getattr(xgb_model, "classes_", np.array([0, 1, 2]))

    def _proba_for(proba_vec: np.ndarray, cls_id: int) -> float:
        try:
            idx_arr = np.where(classes == cls_id)[0]
            if len(idx_arr) == 0:
                return 0.0
            return float(proba_vec[int(idx_arr[0])])
        except Exception:
            return 0.0

    examples = []

    for _, row in latest.iterrows():
        business_name = str(row.get("name", "")).strip()
        neigh_label = str(row.get("analysis_neighborhood", "")).strip()
        insp_dt_hist = row["inspection_date_parsed"].date()
        insp_type_clean = str(row.get("inspection_type_clean", "routine") or "routine").strip().lower()

        if not business_name or not neigh_label:
            continue

        analysis_neighborhood_feature = f"analysis_neighborhood_{neigh_label}"

        # Target inspection date as seen by the model
        target_dt = inspection_date or insp_dt_hist

        payload = PredictRequest(
            business_name=business_name,
            inspection_type=insp_type_clean,
            inspection_date=target_dt,
            analysis_neighborhood=analysis_neighborhood_feature,
            violation_count=violation_count,
        )

        try:
            X_raw = _build_multiclass_features_from_visualization(payload)
            X_scaled = scaler_mc.transform(X_raw)
            proba_vec = xgb_model.predict_proba(X_scaled)[0]
            y_hat = int(xgb_model.predict(X_scaled)[0])
        except Exception:
            continue

        p_fail = _proba_for(proba_vec, 2)
        p_pass = _proba_for(proba_vec, 0)
        p_cond = _proba_for(proba_vec, 1)

        if y_hat == 1 and p_cond >= float(threshold):
            examples.append(
                {
                    "business_name": business_name,
                    "analysis_neighborhood_input": analysis_neighborhood_feature,
                    "analysis_neighborhood_label": neigh_label,
                    "inspection_type_input": insp_type_clean,
                    "inspection_date_input": target_dt.isoformat(),
                    "violation_count_input": violation_count,
                    "proba_pass": p_pass,
                    "proba_conditional": p_cond,
                    "proba_fail": p_fail,
                    "predicted_label": y_hat,
                }
            )
            if len(examples) >= max(1, int(limit)):
                break

    return {
        "threshold": float(threshold),
        "violation_count": violation_count,
        "inspection_date": inspection_date.isoformat() if inspection_date else None,
        "examples": examples,
    }
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
    import numpy as np

    try:
        xgb_metrics = joblib.load(METRICS_PKL_PATH)
    except Exception as exc:
        return {"error": f"Unable to load XGBoost metrics file: {exc}"}

    # Random Forest metrics are optional; if missing, we still return XGBoost
    try:
        rf_metrics = joblib.load(RF_METRICS_PKL_PATH)
    except Exception:
        rf_metrics = {}
    
    # XGBoost: prefer weighted ROC-AUC if available
    weighted_auc_xgb = xgb_metrics.get("roc_auc_weighted")
    try:
        roc_auc_xgb = float(weighted_auc_xgb) if weighted_auc_xgb is not None else None
    except Exception:
        roc_auc_xgb = None

    # RF: weighted AUC and class-2 AUC (accept multiple key names)
    rf_auc_weighted = rf_metrics.get("roc_auc_weighted") or rf_metrics.get("roc_auc")
    try:
        rf_auc_weighted = float(rf_auc_weighted) if rf_auc_weighted is not None else None
    except Exception:
        rf_auc_weighted = None
    
    rf_auc_class2 = None
    try:
        # prefer per-class value if available
        per_class = rf_metrics.get("per_class") or rf_metrics.get("per-class") or {}
        if isinstance(per_class, dict) and 2 in per_class:
            rf_auc_class2 = (
                per_class[2].get("roc_auc")
                or per_class[2].get("roc_auc_score")
            )
        # fallback to explicit key
        if rf_auc_class2 is None:
            rf_auc_class2 = (
                rf_metrics.get("roc_auc_class2")
                or rf_metrics.get("roc_auc_c2")
            )
        rf_auc_class2 = float(rf_auc_class2) if rf_auc_class2 is not None else None
    except Exception:
        rf_auc_class2 = None

    # Basic sizes if present in the XGBoost metrics
    total_records = int(xgb_metrics.get("val_size") or 0)
    n_features = int(xgb_metrics.get("n_features") or 0)
    xgb_auc_class2 = None
    try:
        per_class_xgb = xgb_metrics.get("roc_auc_per_class") or {}
    # Depending on how you saved it:
        val = per_class_xgb.get(2) or per_class_xgb.get("2")
        xgb_auc_class2 = float(val) if val is not None else None
    except Exception:
        xgb_auc_class2 = None
    # Helper to convert any array-like to list[float]
    def _to_float_list(obj):
        try:
            if obj is None:
                return None
            if isinstance(obj, (list, tuple)):
                return [float(v) for v in obj]
            if hasattr(obj, "tolist"):
                try:
                    return [float(v) for v in obj.tolist()]
                except Exception:
                    pass
            return [float(v) for v in list(obj)]
        except Exception:
            return None

    # --- Extract XGBoost class-2 ROC ---
    xgb_c2_fpr = None
    xgb_c2_tpr = None
    try:
        roc_curves_xgb = xgb_metrics.get("roc_curves") or {}
        if isinstance(roc_curves_xgb, dict):
            fpr_dict = roc_curves_xgb.get("fpr", {})
            tpr_dict = roc_curves_xgb.get("tpr", {})

            def _safe_to_list(val):
                if isinstance(val, np.ndarray):
                    return val.tolist()
                return list(val) if isinstance(val, (list, tuple)) else []

            if isinstance(fpr_dict, dict) and isinstance(tpr_dict, dict):
                fpr_val = fpr_dict.get(2) or fpr_dict.get("2")
                tpr_val = tpr_dict.get(2) or tpr_dict.get("2")
                xgb_c2_fpr = _to_float_list(_safe_to_list(fpr_val)) if fpr_val is not None else None
                xgb_c2_tpr = _to_float_list(_safe_to_list(tpr_val)) if tpr_val is not None else None
    except Exception:
        xgb_c2_fpr = None
        xgb_c2_tpr = None

    # --- Extract RF class-2 ROC directly from per_class[2] ---
    rf_c2_fpr = None
    rf_c2_tpr = None
    try:
        per_class_rf = rf_metrics.get("per_class") or rf_metrics.get("per-class") or {}
        pc2 = per_class_rf.get(2) or per_class_rf.get("2")
        if isinstance(pc2, dict):
            pf = pc2.get("fpr")
            pt = pc2.get("tpr")
            if pf is not None and pt is not None:
                rf_c2_fpr = _to_float_list(pf)
                rf_c2_tpr = _to_float_list(pt)
    except Exception:
        rf_c2_fpr = None
        rf_c2_tpr = None

    # --- Build model_roc_curves (raw per-model class-2 curves) ---
    xgb_curve_c2 = None
    if (
        xgb_c2_fpr is not None
        and xgb_c2_tpr is not None
        and len(xgb_c2_fpr) > 0
        and len(xgb_c2_tpr) > 0
    ):
        xgb_curve_c2 = {
            "fpr": [float(v) for v in xgb_c2_fpr],
            "tpr": [float(v) for v in xgb_c2_tpr],
        }

    rf_curve_original = None
    if (
        rf_c2_fpr is not None
        and rf_c2_tpr is not None
        and len(rf_c2_fpr) > 0
        and len(rf_c2_tpr) > 0
    ):
        rf_curve_original = {
            "fpr": [float(v) for v in rf_c2_fpr],
            "tpr": [float(v) for v in rf_c2_tpr],
        }

    model_roc_curves = {
        "xgb_class2": xgb_curve_c2,
        "rf_class2": rf_curve_original,
    }
    featureimportance = xgb_metrics.get("featureimportance")
    top_features = None
    if isinstance(featureimportance, dict):
    # sort descending by importance
        items = sorted(featureimportance.items(), key=lambda kv: kv[1], reverse=True)
        topn = items[:15]
        top_features = [
            {"feature": name, "importance": float(imp)}
        for name, imp in topn
    ]


    # --- Build aligned roc_curves payload for frontend chart ---
    roc_curves_payload = None
    try:
        if (
            xgb_c2_fpr is not None
            and xgb_c2_tpr is not None
            and len(xgb_c2_fpr) > 0
            and len(xgb_c2_tpr) > 0
        ):
            fpr_c2_py = [float(v) for v in xgb_c2_fpr]
            tpr_c2_xgb_py = [float(v) for v in xgb_c2_tpr]
            payload = {"fpr": fpr_c2_py, "tpr_xgb": tpr_c2_xgb_py}

            # If we have RF, align its TPR to the same FPR grid
            if (
                rf_c2_fpr is not None
                and rf_c2_tpr is not None
                and len(rf_c2_fpr) > 0
                and len(rf_c2_tpr) > 0
            ):
                if len(rf_c2_fpr) == len(fpr_c2_py):
                    payload["tpr_rf"] = [float(v) for v in rf_c2_tpr]
                elif len(rf_c2_fpr) >= 2:
                    try:
                        xf = np.asarray(fpr_c2_py, dtype=float)
                        rf_x = np.asarray(rf_c2_fpr, dtype=float)
                        rf_y = np.asarray(rf_c2_tpr, dtype=float)
                        sorted_idx = np.argsort(rf_x)
                        rf_x_sorted = rf_x[sorted_idx]
                        rf_y_sorted = rf_y[sorted_idx]
                        ux, inv_idx = np.unique(rf_x_sorted, return_index=True)
                        if len(ux) >= 2:
                            rf_x_unique = rf_x_sorted[inv_idx]
                            rf_y_unique = rf_y_sorted[inv_idx]
                            interp_y = np.interp(
                                xf,
                                rf_x_unique,
                                rf_y_unique,
                                left=0.0,
                                right=1.0,
                            )
                            payload["tpr_rf"] = interp_y.astype(float).tolist()
                        else:
                            payload["tpr_rf"] = [
                                float(rf_y_sorted[0]) if len(rf_y_sorted) > 0 else 0.0
                                for _ in xf
                            ]
                    except Exception:
                        # if interpolation fails, just skip RF in aligned payload
                        pass

            roc_curves_payload = payload

    except Exception:
        roc_curves_payload = None

    # Fallback: if aligned payload is still None but we have xgb_class2, at least expose that
    if roc_curves_payload is None and xgb_curve_c2 is not None:
        roc_curves_payload = {
            "fpr": xgb_curve_c2["fpr"],
            "tpr_xgb": xgb_curve_c2["tpr"],
            "tpr_rf": None,
        }
    cal = xgb_metrics.get("calibration") or xgb_metrics.get("calibration_curve")
    calibration = None
    if cal is not None and isinstance(cal, dict):
        # Expect something like {"bin_center": [...], "observed": [...]}
        xs = cal.get("bin_center") or cal.get("pred_bin") or []
        ys = cal.get("observed") or cal.get("empirical") or []
        if len(xs) and len(xs) == len(ys):
            calibration = [
                {"pred_bin": float(x), "observed": float(y)}
                for x, y in zip(xs, ys)
        ]
    # payload["calibrationCurve"] = calibration
    corr_vals = xgb_metrics.get("correlations")
    corr_feats = xgb_metrics.get("features")
    correlations = None
    if isinstance(corr_vals, (list, tuple)) and isinstance(corr_feats, (list, tuple)):
        correlations = [
            {"feature": f, "correlation": float(c)}
            for f, c in zip(corr_feats, corr_vals)
        ]
    # payload["correlations"] = correlations

    featureimportance = xgb_metrics.get("featureimportance")
    top_features = None
    if isinstance(featureimportance, dict):
        # sort descending by importance
        items = sorted(featureimportance.items(), key=lambda kv: kv[1], reverse=True)
        topn = items[:10]  # <-- only top 10
        top_features = [
            {"feature": name, "importance": float(imp)}
            for name, imp in topn
        ]
    # --- Random Forest metrics extraction ---
    rf_classification_report = None
    rf_confusion_matrix = None
    rf_feature_importance = None
    rf_metrics_dict = {}
    try:
        # Try to extract classification_report, confusion_matrix, feature_importance, and key metrics
        if isinstance(rf_metrics, dict):
            rf_classification_report = rf_metrics.get("classification_report")
            rf_confusion_matrix = rf_metrics.get("confusion_matrix")
            rf_feature_importance = None
            rf_feat_imp = rf_metrics.get("featureimportance")
            if isinstance(rf_feat_imp, dict):
                items = sorted(rf_feat_imp.items(), key=lambda kv: kv[1], reverse=True)
                topn = items[:10]
                rf_feature_importance = [
                    {"feature": name, "importance": float(imp)} for name, imp in topn
                ]
            # Key metrics
            for k in [
                "accuracy", "macro_f1", "weighted_f1", "macro_precision", "weighted_precision", "macro_recall", "weighted_recall"
            ]:
                v = rf_metrics.get(k)
                if v is not None:
                    rf_metrics_dict[k] = float(v)
    except Exception:
        pass

    # --- XGBoost metrics extraction ---
    xgb_classification_report = xgb_metrics.get("classification_report")
    xgb_confusion_matrix = xgb_metrics.get("confusion_matrix")
    xgb_feature_importance = None
    xgb_feat_imp = xgb_metrics.get("featureimportance")
    if isinstance(xgb_feat_imp, dict):
        items = sorted(xgb_feat_imp.items(), key=lambda kv: kv[1], reverse=True)
        topn = items[:10]
        xgb_feature_importance = [
            {"feature": name, "importance": float(imp)} for name, imp in topn
        ]
    xgb_metrics_dict = {}
    for k in [
        "accuracy", "macro_f1", "weighted_f1", "macro_precision", "weighted_precision", "macro_recall", "weighted_recall"
    ]:
        v = xgb_metrics.get(k)
        if v is not None:
            xgb_metrics_dict[k] = float(v)

    # Return the full insights payload (keep existing keys)
    return {
        "total_records": total_records,
        "models": [],
        "roc_curves": roc_curves_payload,
        "rf_curve": rf_curve_original,
        "model_roc_curves": model_roc_curves,
        "correlations": {"features": [], "values": []},
        "fail_rate_by_year": [],
        "xg_roc_auc_weighted": roc_auc_xgb,
        "xg_roc_auc_class2": xgb_auc_class2,
        "rf_roc_auc_class2": rf_auc_class2,
        "rf_roc_auc_weighted": rf_auc_weighted,
        "recall_fail": None,
        "n_features": n_features,
        "classification_report": xgb_classification_report,
        "confusion_matrix": xgb_confusion_matrix,
        "feature_importance": xgb_feature_importance,
        "featureImportance": top_features,
        "calibrationCurve": calibration,
        "topFeatures": top_features,
        "xgb_metrics": xgb_metrics_dict,
        # --- Random Forest additions ---
        "rf_classification_report": rf_classification_report,
        "rf_confusion_matrix": rf_confusion_matrix,
        "rf_feature_importance": rf_feature_importance,
        "rf_metrics": rf_metrics_dict,
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
