import json
import math
import lightgbm as lgb
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------
# Load artifacts
# -----------------------
with open("artifacts/mileage_priors.json") as f:
    PRIORS = json.load(f)

MODEL = lgb.Booster(model_file="artifacts/residual_lgbm.txt")

# -----------------------
# Trim tier (same as training)
# -----------------------
def trim_tier(t):
    t = str(t).upper()
    if any(x in t for x in ["DENALI", "PLATINUM", "HIGH COUNTRY", "LIMITED"]):
        return 3
    if any(x in t for x in ["LARIAT", "LTZ", "SLT", "REBEL"]):
        return 2
    return 1

# -----------------------
# API schema
# -----------------------
class Request(BaseModel):
    est_value: float
    year: int
    odometer: int
    make: str
    model: str
    trim: str

app = FastAPI(title="Truck Price Correction API")

# -----------------------
# Blend logic
# -----------------------
def blend_alpha(residual, prior):
    ratio = abs(residual) / max(prior, 1)
    if ratio > 0.2:
        return 0.2
    if ratio > 0.1:
        return 0.5
    return 1.0

# -----------------------
# Endpoint
# -----------------------
@app.post("/correct")
def correct_price(r: Request):
    key = f"{r.make.upper()}_{r.model.upper()}"

    if key not in PRIORS:
        return {"applied": False, "reason": "no_prior"}

    p = PRIORS[key]
    prior_price = p["slope"] * math.log1p(r.odometer) + p["intercept"]

    X = [[
        r.est_value,
        r.odometer,
        r.year,
        trim_tier(r.trim)
    ]]

    residual = float(MODEL.predict(X)[0])
    alpha = blend_alpha(residual, prior_price)

    final_price = prior_price + alpha * residual

    return {
        "applied": True,
        "prior_price": round(prior_price, 2),
        "residual_raw": round(residual, 2),
        "alpha": alpha,
        "final_price": round(final_price, 2)
    }