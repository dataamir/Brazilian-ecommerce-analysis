# ============================================================
# notebook_03_customer_segmentation.py
#
# Step 3 of the project:
#   - Load the master table
#   - Build the RFM table (one row per customer)
#   - Score each customer 1-5 on R, F, M
#   - Label segments: Champions, Loyal, Recent, At Risk, Lost
#   - Visualise segment distribution and avg spend
#   - Find the one-time buyer problem
#   - Save rfm_segments.csv
#
# Run notebook_01 first to generate the master table.
# ============================================================

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("./src")
from rfm_model      import run_rfm
from visualizations import plot_rfm_segments

os.makedirs("./reports", exist_ok=True)


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
print("Loading master table...")
master = pd.read_parquet("./data/processed/master_table.parquet")
print(f"Loaded: {master.shape[0]:,} rows")


# ----------------------------------------------------------
# 1. BUILD RFM TABLE
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 1 — BUILD RFM TABLE")
print("=" * 50)

rfm = run_rfm(master)

print("\nFirst 5 rows of RFM table:")
print(rfm.head())


# ----------------------------------------------------------
# 2. SEGMENT DISTRIBUTION CHART
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 2 — VISUALISE SEGMENTS")
print("=" * 50)

fig = plot_rfm_segments(rfm)
fig.savefig("./reports/fig_07_rfm_segments.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/fig_07_rfm_segments.png")


# ----------------------------------------------------------
# 3. SEGMENT PROFILES
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 3 — SEGMENT PROFILES")
print("=" * 50)

profile = (
    rfm.groupby("Segment")
    .agg(
        customers      = ("customer_unique_id", "count"),
        avg_recency    = ("Recency",   "mean"),
        avg_frequency  = ("Frequency", "mean"),
        avg_spend      = ("Monetary",  "mean"),
        total_revenue  = ("Monetary",  "sum"),
    )
    .round(1)
)

# add percentage columns
profile["pct_customers"] = (profile["customers"] / profile["customers"].sum() * 100).round(1)
profile["pct_revenue"]   = (profile["total_revenue"] / profile["total_revenue"].sum() * 100).round(1)

print(profile.sort_values("total_revenue", ascending=False).to_string())


# ----------------------------------------------------------
# 4. RFM DISTRIBUTION HISTOGRAMS
# ----------------------------------------------------------
print("\nPlotting RFM distributions...")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

configs = [
    ("Recency",   "Days Since Last Order", "#2563a8"),
    ("Frequency", "Number of Orders",      "#c8401a"),
    ("Monetary",  "Total Spend (BRL)",     "#1a6b3c"),
]

for ax, (col, title, color) in zip(axes, configs):
    # clip at 95th percentile so chart isn't squashed by extreme values
    data = rfm[col].clip(upper=rfm[col].quantile(0.95))
    ax.hist(data, bins=30, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Customers")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.suptitle("RFM Value Distributions (clipped at 95th percentile)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig("./reports/fig_08_rfm_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/fig_08_rfm_distributions.png")


# ----------------------------------------------------------
# 5. ONE-TIME BUYER PROBLEM
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 4 — ONE-TIME BUYER ANALYSIS")
print("=" * 50)

one_time = (rfm["Frequency"] == 1).sum()
total    = len(rfm)

print(f"Customers who only bought once: {one_time:,} ({one_time/total*100:.1f}%)")
print(f"Customers who bought 2+ times:  {total-one_time:,} ({(total-one_time)/total*100:.1f}%)")

print("\nFrequency breakdown:")
freq_dist = rfm["Frequency"].value_counts().head(6)
for freq, count in freq_dist.items():
    bar = "█" * int(count / total * 200)
    print(f"  Bought {freq:2d}x : {count:6,} ({count/total*100:5.1f}%)  {bar}")


# ----------------------------------------------------------
# 6. CHAMPIONS DEEP DIVE
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 5 — CHAMPIONS SEGMENT DEEP DIVE")
print("=" * 50)

champions = rfm[rfm["Segment"] == "Champions"]
print(f"Champions count:    {len(champions):,} customers")
print(f"Champions revenue:  R${champions['Monetary'].sum():,.2f}")
print(f"Avg champion spend: R${champions['Monetary'].mean():.2f}")
print(f"Avg recency:        {champions['Recency'].mean():.0f} days")

champ_rev_share = champions["Monetary"].sum() / rfm["Monetary"].sum() * 100
print(f"\nChampions are {len(champions)/total*100:.1f}% of customers")
print(f"but generate {champ_rev_share:.1f}% of total revenue")


# ----------------------------------------------------------
# 7. SAVE
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 6 — SAVE RFM TABLE")
print("=" * 50)

rfm.to_csv("./data/processed/rfm_segments.csv", index=False)
print(f"Saved: data/processed/rfm_segments.csv ({len(rfm):,} customers)")

print("\nNotebook 3 complete!")
