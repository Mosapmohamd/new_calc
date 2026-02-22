import json
import lightgbm as lgb
import pandas as pd
from pathlib import Path

ART = Path("artifacts")

encoder = json.load(open(ART / "encoder.json"))

models = {
    10: lgb.Booster(model_file=str(ART / "model_q10.txt")),
    50: lgb.Booster(model_file=str(ART / "model_q50.txt")),
    90: lgb.Booster(model_file=str(ART / "model_q90.txt")),
}

def encode_input(x):
    key = f"{x['make'].upper()}_{x['segment']}"
    if key not in encoder:
        raise ValueError(f"Unknown make_model: {key}")

    return pd.DataFrame([{
        "year": x["year"],
        "odometer": x["odometer"],
        "est_value": x["est_value"],
        "make_model_enc": encoder[key],
        "trim_tier": x["trim_tier"],
    }])

def predict(x):
    X = encode_input(x)
    preds = {q: models[q].predict(X)[0] for q in models}
    final = x["est_value"] + preds[50]

    return {
        "p10": x["est_value"] + preds[10],
        "p50": final,
        "p90": x["est_value"] + preds[90],
        "applied": True,
    }
