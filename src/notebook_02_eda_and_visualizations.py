# ============================================================
# notebook_02_eda_and_visualizations.py
#
# Step 2 of the project:
#   - Load the cleaned master table
#   - Print key summary statistics
#   - Plot monthly revenue trend
#   - Plot order heatmap (day x hour)
#   - Plot payment method breakdown
#   - Plot top 10 categories by revenue
#   - Plot review score vs delivery timing
#   - Plot orders by state
#
# Run notebook_01 first to generate the master table.
# ============================================================

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("./src")
from visualizations import (
    plot_monthly_revenue,
    plot_review_vs_delay,
    plot_order_heatmap,
    plot_top_categories,
)

os.makedirs("./reports", exist_ok=True)


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
print("Loading master table...")
master = pd.read_parquet("./data/processed/master_table.parquet")
print(f"Loaded: {master.shape[0]:,} rows x {master.shape[1]} columns")


# ----------------------------------------------------------
# 1. KEY SUMMARY NUMBERS
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("KEY NUMBERS")
print("=" * 50)

print(f"Total orders:        {master['order_id'].nunique():,}")
print(f"Total customers:     {master['customer_unique_id'].nunique():,}")
print(f"Total revenue:       R${master['total_value'].sum():,.2f}")
print(f"Avg order value:     R${master['total_value'].mean():.2f}")
print(f"Avg delivery days:   {master['delivery_days'].mean():.1f}")
print(f"Avg review score:    {master['review_score'].mean():.2f} stars")
print(f"Late delivery rate:  {master['is_late'].mean() * 100:.1f}%")


# ----------------------------------------------------------
# 2. MONTHLY REVENUE TREND
# ----------------------------------------------------------
print("\nPlotting monthly revenue...")

fig = plot_monthly_revenue(master)
fig.savefig("./reports/fig_01_monthly_revenue.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/fig_01_monthly_revenue.png")

# print the monthly numbers too
monthly = master.groupby("order_month")["total_value"].sum()
print("\nMonthly revenue (R$):")
print(monthly.tail(6).round(2))


# ----------------------------------------------------------
# 3. ORDER HEATMAP — DAY x HOUR
# ----------------------------------------------------------
print("\nPlotting order heatmap...")

fig = plot_order_heatmap(master)
fig.savefig("./reports/fig_02_order_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/fig_02_order_heatmap.png")

# busiest hour and day
print(f"\nBusiest hour: {master.groupby('order_hour')['order_id'].count().idxmax()}:00")
print(f"Busiest day:  {master.groupby('order_dayofweek')['order_id'].count().idxmax()}")


# ----------------------------------------------------------
# 4. PAYMENT METHODS
# ----------------------------------------------------------
print("\nPlotting payment methods...")

# pull from SQLite using SQL query
conn = sqlite3.connect("./data/olist.db")
payments_df = pd.read_sql("""
    SELECT
        payment_type,
        COUNT(DISTINCT order_id) AS orders,
        ROUND(SUM(payment_value), 2) AS total_value,
        ROUND(AVG(payment_value), 2)  AS avg_value
    FROM payments
    GROUP BY payment_type
    ORDER BY orders DESC
""", conn)
conn.close()

print("\nPayment method breakdown:")
print(payments_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
bar_colors = ["#2563a8", "#c8401a", "#d4a017", "#1a6b3c"]
bars = ax.bar(payments_df["payment_type"], payments_df["orders"],
              color=bar_colors, edgecolor="white")
for bar, val in zip(bars, payments_df["orders"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{val:,}", ha="center", fontsize=9)
ax.set_title("Orders by Payment Method", fontsize=12, fontweight="bold")
ax.set_ylabel("Number of Orders")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig("./reports/fig_03_payment_methods.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/fig_03_payment_methods.png")


# ----------------------------------------------------------
# 5. TOP 10 CATEGORIES BY REVENUE
# ----------------------------------------------------------
print("\nPlotting top categories...")

fig = plot_top_categories(master, top_n=10)
fig.savefig("./reports/fig_04_top_categories.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/fig_04_top_categories.png")


# ----------------------------------------------------------
# 6. REVIEW SCORE VS DELIVERY TIMING
# ----------------------------------------------------------
print("\nPlotting review score vs delivery timing...")

fig = plot_review_vs_delay(master)
fig.savefig("./reports/fig_05_review_vs_delay.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/fig_05_review_vs_delay.png")

# key numbers
on_time_score = master[~master["is_late"]]["review_score"].mean()
late_score    = master[master["is_late"]]["review_score"].mean()
print(f"\nOn-time delivery avg score: {on_time_score:.2f} stars")
print(f"Late delivery avg score:    {late_score:.2f} stars")
print(f"Difference: {on_time_score - late_score:.2f} stars drop")


# ----------------------------------------------------------
# 7. ORDERS BY STATE
# ----------------------------------------------------------
print("\nPlotting orders by state...")

state_orders = (
    master.groupby("customer_state")["order_id"]
    .nunique()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(state_orders.index, state_orders.values, color="#2563a8", edgecolor="white")
ax.set_title("Orders by Customer State", fontsize=12, fontweight="bold")
ax.set_xlabel("State")
ax.set_ylabel("Number of Orders")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig("./reports/fig_06_orders_by_state.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: reports/fig_06_orders_by_state.png")

sp_share = state_orders["SP"] / state_orders.sum() * 100
print(f"\nTop state: SP with {sp_share:.1f}% of all orders")
print(f"Top 3 states: {list(state_orders.head(3).index)}")

print("\nNotebook 2 complete — all charts saved to ./reports/")
