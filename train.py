import os
import json
import math
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------
# Config
# -----------------------
DATA_PATH = "data/trucks_clean.csv"
ARTIFACTS = "artifacts"
os.makedirs(ARTIFACTS, exist_ok=True)

MIN_CURVE_ROWS = 15

# -----------------------
# Load
# -----------------------
df = pd.read_csv(DATA_PATH)

required = ["make", "model", "trim", "year", "odometer", "est_value", "real_value"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["make_model"] = df["make"].str.upper() + "_" + df["model"].str.upper()

# -----------------------
# 1) Trim tier encoding
# -----------------------
def trim_tier(t):
    t = str(t).upper()
    if any(x in t for x in ["DENALI", "PLATINUM", "HIGH COUNTRY", "LIMITED"]):
        return 3
    if any(x in t for x in ["LARIAT", "LTZ", "SLT", "REBEL"]):
        return 2
    return 1

df["trim_tier"] = df["trim"].apply(trim_tier)

# -----------------------
# 2) Mileage priors
# -----------------------
priors = {}

for mm, g in df.groupby("make_model"):
    if len(g) < MIN_CURVE_ROWS:
        continue

    x = np.log1p(g["odometer"].values)
    y = g["real_value"].values

    slope, intercept = np.polyfit(x, y, 1)
    priors[mm] = {
        "slope": float(slope),
        "intercept": float(intercept),
        "n": int(len(g))
    }

with open(f"{ARTIFACTS}/mileage_priors.json", "w") as f:
    json.dump(priors, f, indent=2)

print(f"Saved {len(priors)} mileage priors")

# -----------------------
# 3) Prior + residual
# -----------------------
def prior_price(row):
    mm = row["make_model"]
    if mm not in priors:
        return row["est_value"]
    p = priors[mm]
    return p["slope"] * math.log1p(row["odometer"]) + p["intercept"]

df["prior_price"] = df.apply(prior_price, axis=1)
df["residual"] = df["real_value"] - df["prior_price"]

# -----------------------
# 4) Features
# -----------------------
FEATURES = [
    "est_value",
    "odometer",
    "year",
    "trim_tier"
]

X = df[FEATURES]
y = df["residual"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# -----------------------
# 5) Train residual model
# -----------------------
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.04,
    max_depth=6,
    min_data_in_leaf=12,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------
# 6) Evaluation
# -----------------------
baseline_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
preds = model.predict(X_test)
residual_mae = mean_absolute_error(y_test, preds)

print("\nRESIDUAL BASELINE MAE:", round(baseline_mae, 2))
print("RESIDUAL MODEL MAE:", round(residual_mae, 2))

# -----------------------
# 7) Save model
# -----------------------
model.booster_.save_model(f"{ARTIFACTS}/residual_lgbm.txt")

print("\nArtifacts saved:")
print("- mileage_priors.json")
print("- residual_lgbm.txt")