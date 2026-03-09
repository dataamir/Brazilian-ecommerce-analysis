# ============================================================
# notebook_01_data_loading_and_cleaning.py
#
# Step 1 of the project:
#   - Load all 9 Olist CSV files
#   - Audit for missing values and wrong data types
#   - Clean the orders and order_items tables
#   - Join everything into one master table
#   - Add derived features
#   - Save the cleaned data for the next notebooks
# ============================================================

import os
import sys
import pandas as pd
import numpy as np

# add src folder so we can import our helper files
sys.path.append("./src")

from load_data  import load_all_tables, build_master_table, save_to_sqlite
from clean_data import clean_orders, clean_order_items, add_features, null_report


# ----------------------------------------------------------
# 1. LOAD ALL CSV FILES
# ----------------------------------------------------------
print("=" * 50)
print("STEP 1 — LOADING DATA")
print("=" * 50)

# make sure data/raw/ folder has the Kaggle CSVs
# download command:  kaggle datasets download -d olistbr/brazilian-ecommerce -p ./data/raw --unzip

tables = load_all_tables("./data/raw/")

# print shape of every table
print("\nTable shapes:")
for name, df in tables.items():
    print(f"  {name:20s}: {df.shape[0]:>8,} rows  x  {df.shape[1]:>2} cols")


# ----------------------------------------------------------
# 2. AUDIT — CHECK FOR PROBLEMS
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 2 — NULL AUDIT")
print("=" * 50)

print("\nOrders table dtypes:")
print(tables["orders"].dtypes)

print("\nNull report for orders:")
null_report(tables["orders"])

print("\nNull report for order_items:")
null_report(tables["order_items"])

print("\nOrder status breakdown:")
print(tables["orders"]["order_status"].value_counts())


# ----------------------------------------------------------
# 3. CLEAN ORDERS
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 3 — CLEANING ORDERS")
print("=" * 50)

tables["orders"] = clean_orders(tables["orders"])


# ----------------------------------------------------------
# 4. CLEAN ORDER ITEMS
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 4 — CLEANING ORDER ITEMS")
print("=" * 50)

tables["order_items"] = clean_order_items(tables["order_items"])

print("\nPrice stats after cleaning:")
print(tables["order_items"]["price"].describe().round(2))


# ----------------------------------------------------------
# 5. BUILD MASTER TABLE
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 5 — BUILD MASTER TABLE")
print("=" * 50)

master = build_master_table(tables)

print("\nFirst 3 rows:")
print(master.head(3))

print("\nColumns in master table:")
print(list(master.columns))


# ----------------------------------------------------------
# 6. ADD FEATURES
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 6 — ADD DERIVED FEATURES")
print("=" * 50)

master = add_features(master)

# check the new columns
print("\nNew feature sample:")
print(master[["total_value", "delivery_days", "delay_days", "is_late", "order_hour"]].head(5))

# quick stats
print(f"\nAvg delivery time:  {master['delivery_days'].mean():.1f} days")
print(f"Avg total value:    R${master['total_value'].mean():.2f}")
print(f"Avg review score:   {master['review_score'].mean():.2f}")
print(f"Date range: {master['order_purchase_timestamp'].min().date()} to {master['order_purchase_timestamp'].max().date()}")


# ----------------------------------------------------------
# 7. SAVE CLEANED DATA
# ----------------------------------------------------------
print("\n" + "=" * 50)
print("STEP 7 — SAVE")
print("=" * 50)

os.makedirs("./data/processed", exist_ok=True)

# save as parquet (compact format, fast to reload)
master.to_parquet("./data/processed/master_table.parquet", index=False)
print("Saved: data/processed/master_table.parquet")

# also save to SQLite so we can run SQL queries later
save_to_sqlite(tables, db_path="./data/olist.db")

print("\nNotebook 1 complete!")
print(f"Master table shape: {master.shape}")
