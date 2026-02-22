import pandas as pd
import numpy as np

# ---- Trim tier rules (generic, extensible) ----
TRIM_TIERS = {
    "luxury": ["denali", "platinum", "limited", "high country", "lariat"],
    "offroad": ["trx", "raptor", "at4", "zr2", "rebel"],
    "mid": ["sl", "lt", "xlt", "slt", "big horn"],
}

def encode_trim_tiers(trim: str):
    trim = str(trim).lower()
    out = {
        "trim_luxury": 0,
        "trim_offroad": 0,
        "trim_mid": 0,
    }
    for tier, keywords in TRIM_TIERS.items():
        if any(k in trim for k in keywords):
            out[f"trim_{tier}"] = 1
    return out

def make_model_key(make, model):
    return f"{str(make).lower()}__{str(model).lower()}"

def build_features(df: pd.DataFrame, training=True):
    df = df.copy()

    # ---- Core numeric features ----
    df["age"] = 2026 - df["year"]
    df["log_odometer"] = np.log1p(df["odometer"])
    df["log_est"] = np.log1p(df["est_value"])

    # ---- Make-model interaction ----
    df["make_model"] = df.apply(
        lambda r: make_model_key(r["make"], r["model"]), axis=1
    )

    # ---- Trim tiers ----
    trim_df = df["trim"].apply(encode_trim_tiers).apply(pd.Series)
    df = pd.concat([df, trim_df], axis=1)

    features = [
        "log_est",
        "age",
        "log_odometer",
        "trim_luxury",
        "trim_offroad",
        "trim_mid",
    ]

    return df, features
