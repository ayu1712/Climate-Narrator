import pandas as pd
import numpy as np
import os

df = pd.read_parquet(os.path.join('data', 'processed', 'raw_combined.parquet'))

# ── Filter to match paper: 2005–2024, valid narratives, non-zero damage ────────
# ── Parse NOAA damage strings like "1.5K", "2M", "0", "" ─────────────────────
def parse_damage(val):
    if pd.isna(val):
        return 0.0
    s = str(val).strip().upper().replace(",", "")
    if s in ("", "0", "0.0"):
        return 0.0
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "H": 1e2, "T": 1e12}
    if s[-1] in multipliers:
        try:
            return float(s[:-1]) * multipliers[s[-1]]
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0
 
print("Parsing damage columns...")
df["DAMAGE_PROPERTY_NUM"] = df["DAMAGE_PROPERTY"].apply(parse_damage)
df["DAMAGE_CROPS_NUM"]    = df["DAMAGE_CROPS"].apply(parse_damage)
df["total_damage"]        = df["DAMAGE_PROPERTY_NUM"] + df["DAMAGE_CROPS_NUM"]

# Combine narrative fields (adjust column names if needed)
df["narrative"] = (
    df["EVENT_NARRATIVE"].fillna("") + " " + df["EPISODE_NARRATIVE"].fillna("")
).str.strip()

df["narrative_word_count"] = df["narrative"].str.split().str.len()

# Apply the same filters used in training
filtered = df[
    (df["narrative_word_count"] >= 20) &
    (df["total_damage"] > 0)
].copy()

# ── Stats ──────────────────────────────────────────────────────────────────────
raw_count        = len(df)
filtered_count   = len(filtered)
avg_narrative    = filtered["narrative_word_count"].mean()
avg_damage       = filtered["total_damage"].mean()
max_damage       = filtered["total_damage"].max()
fatality_events  = (filtered["DEATHS_DIRECT"] + filtered["DEATHS_INDIRECT"] > 0).sum()

# Temporal split counts
train = filtered[filtered["YEAR"] < 2019]
val   = filtered[(filtered["YEAR"] >= 2019) & (filtered["YEAR"] <= 2021)]
test  = filtered[filtered["YEAR"] >= 2022]

print("=" * 45)
print(f"Raw events:              {raw_count:>12,}")
print(f"Filtered events:         {filtered_count:>12,}")
print(f"Training set (2005-2018):{len(train):>12,}")
print(f"Validation (2019-2021):  {len(val):>12,}")
print(f"Test set (2022-2024):    {len(test):>12,}")
print("-" * 45)
print(f"Avg narrative length:    {avg_narrative:>11.0f} words")
print(f"Avg property damage:     ${avg_damage/1e6:>10.2f}M")
print(f"Max property damage:     ${max_damage/1e9:>10.2f}B")
print(f"Events with fatalities:  {fatality_events:>12,}")
print("=" * 45)
