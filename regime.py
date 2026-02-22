import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def train_market_regime(df):
    X = df[["year", "real_value", "odometer"]].copy()
    X["log_price"] = np.log1p(X["real_value"])
    X["log_odo"] = np.log1p(X["odometer"])

    km = KMeans(n_clusters=3, random_state=42)
    labels = km.fit_predict(X[["year", "log_price", "log_odo"]])

    centers = km.cluster_centers_

    return {
        "centers": centers.tolist()
    }
