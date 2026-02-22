import pandas as pd
import numpy as np

def compute_trim_shrinkage(df):
    stats = {}

    global_mean = df["residual"].mean()
    global_var = df["residual"].var()

    for (make, model, trim), g in df.groupby(["make", "model", "trim"]):
        n = len(g)
        mean = g["residual"].mean()
        var = g["residual"].var() if n > 1 else global_var

        weight = n / (n + 20)  # shrinkage strength
        shrunk_mean = weight * mean + (1 - weight) * global_mean

        stats[f"{make}_{model}_{trim}"] = {
            "mean": float(shrunk_mean),
            "n": n
        }

    return stats
