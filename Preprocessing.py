import pandas as pd
import re

df = pd.read_csv("data/trucks.csv")

# ---------------------------
# Normalize text
# ---------------------------
def norm(s):
    if pd.isna(s):
        return ""
    return re.sub(r"\s+", "", s.upper())

df["make"] = df["make"].str.upper().str.strip()
df["model_raw"] = df["model"].astype(str)
df["model_norm"] = df["model_raw"].apply(norm)

# ---------------------------
# Canonical model mapping
# ---------------------------
def canonical_model(make, model):
    m = model

    # FORD
    if make == "FORD":
        if "F150" in m:
            return "F150"
        if "F250" in m:
            return "F250"
        if "F350" in m:
            return "F350"

    # GMC
    if make == "GMC":
        if "SIERRA1500" in m:
            return "SIERRA_1500"
        if "SIERRA2500" in m:
            return "SIERRA_2500HD"
        if "SIERRA3500" in m:
            return "SIERRA_3500HD"

    # CHEVROLET
    if make == "CHEVROLET":
        if "SILVERADO1500" in m:
            return "SILVERADO_1500"
        if "SILVERADO2500" in m:
            return "SILVERADO_2500HD"
        if "SILVERADO3500" in m:
            return "SILVERADO_3500HD"

    # RAM / DODGE
    if make in ["RAM", "DODGE"]:
        if "1500" in m:
            return "RAM_1500"
        if "2500" in m:
            return "RAM_2500"
        if "3500" in m:
            return "RAM_3500"

    return None

df["model"] = df.apply(lambda r: canonical_model(r["make"], r["model_norm"]), axis=1)

# ---------------------------
# Drop invalid rows
# ---------------------------
before = len(df)
df = df[df["model"].notna()]
after = len(df)

print(f"Dropped {before - after} rows with unknown models")

# ---------------------------
# Numeric cleanup
# ---------------------------
for col in ["odometer", "est_value", "real_value"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["odometer", "est_value", "real_value"])

# ---------------------------
# Residual
# ---------------------------
df["residual"] = df["real_value"] - df["est_value"]

# ---------------------------
# Save clean dataset
# ---------------------------
df.to_csv("data/trucks_clean.csv", index=False)

print("\nFinal model distribution:")
print(df["model"].value_counts())
