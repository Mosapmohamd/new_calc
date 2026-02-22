import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

# ---------------- CONFIG ----------------
DATA_PATH = "data/trucks_clean.csv"
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- LOAD ----------------
df = pd.read_csv(DATA_PATH)

print("Loaded rows:", len(df))

# ======================================================
# 1️⃣ MILEAGE DECAY CURVES (per make-model)
# ======================================================

def build_mileage_curves(df):
    curves = {}

    df = df[df["odometer"] > 0].copy()
    df["bucket"] = (df["odometer"] // 25000) * 25000

    for (make, model), g in df.groupby(["make", "model"]):
        med = g.groupby("bucket")["real_value"].median()
        if len(med) < 3:
            continue

        x = np.log1p(med.index.values)
        y = med.values

        slope, intercept = np.polyfit(x, y, 1)
        curves[f"{make}_{model}"] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "points": int(len(med))
        }

    return curves

mileage_curves = build_mileage_curves(df)

with open(f"{OUT_DIR}/mileage_curves.json", "w") as f:
    json.dump(mileage_curves, f, indent=2)

print("Saved mileage_curves.json")

# ======================================================
# 2️⃣ HIERARCHICAL SHRINKAGE (trim-level)
# ======================================================

def build_shrinkage(df):
    stats = {}
    global_mean = df["residual"].mean()

    for (make, model, trim), g in df.groupby(["make", "model", "trim"]):
        n = len(g)
        mean = g["residual"].mean()

        weight = n / (n + 20)  # shrinkage strength
        shrunk = weight * mean + (1 - weight) * global_mean

        stats[f"{make}_{model}_{trim}"] = {
            "shrunk_mean": float(shrunk),
            "n": int(n)
        }

    return stats

shrinkage_stats = build_shrinkage(df)

with open(f"{OUT_DIR}/shrinkage_stats.json", "w") as f:
    json.dump(shrinkage_stats, f, indent=2)

print("Saved shrinkage_stats.json")

# ======================================================
# 3️⃣ MARKET REGIME CLUSTERING
# ======================================================

def build_market_regime(df):
    X = df[["year", "real_value", "odometer"]].copy()
    X["log_price"] = np.log1p(X["real_value"])
    X["log_odo"] = np.log1p(X["odometer"])

    km = KMeans(n_clusters=3, random_state=42)
    km.fit(X[["year", "log_price", "log_odo"]])

    return {
        "centers": km.cluster_centers_.tolist()
    }

regime_model = build_market_regime(df)

with open(f"{OUT_DIR}/regime_model.json", "w") as f:
    json.dump(regime_model, f, indent=2)

print("Saved regime_model.json")

print("\n✅ ALL PRIORS BUILT SUCCESSFULLY")
