import pandas as pd
import numpy as np
import json
from collections import defaultdict

def build_mileage_curves(df):
    curves = defaultdict(dict)

    df["bucket"] = (df["odometer"] // 25000) * 25000

    for (make, model), g in df.groupby(["make", "model"]):
        medians = g.groupby("bucket")["real_value"].median()
        if len(medians) < 3:
            continue

        x = medians.index.values
        y = medians.values

        coeff = np.polyfit(np.log1p(x), y, 1)
        curves[f"{make}_{model}"] = {
            "slope": float(coeff[0]),
            "intercept": float(coeff[1])
        }

    return curves
