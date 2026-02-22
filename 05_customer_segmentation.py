"""
TFM Farmalisto Analytics
Script: Modelo 3 - Customer segmentation

Objetivo:
- Segmentar clientes en perfiles accionables (marketing/CRM) usando RFM:
  - Recency: días desde última compra
  - Frequency: nº pedidos
  - Monetary: gasto total (payment_value)

"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Farmalisto_TFM
FACT_FP = PROJECT_ROOT / "data_processed" / "fact_orders.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "models" / "model03_customer_segmentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 4
RANDOM_STATE = 42

# usar el valor real del pedido
MONEY_COL = "payment_value"

# winsorization para evitar que 1 cliente raro distorsione clusters
WINSOR_PCT_LOW = 0.01
WINSOR_PCT_HIGH = 0.99


# =========================
# HELPERS
# =========================
def winsorize_series(s: pd.Series, p_low: float, p_high: float) -> pd.Series:
    lo = s.quantile(p_low)
    hi = s.quantile(p_high)
    return s.clip(lower=lo, upper=hi)


# =========================
# LOAD
# =========================
if not FACT_FP.exists():
    raise FileNotFoundError(f"No encuentro: {FACT_FP}")

df = pd.read_csv(FACT_FP)

required = ["order_id", "customer_id", "order_purchase_timestamp", MONEY_COL]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Faltan columnas: {missing}. Disponibles: {sorted(df.columns.tolist())}")

df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
df = df.loc[df["order_purchase_timestamp"].notna()].copy()

# nos quedamos solo con pedidos válidos y con importe
df = df.loc[df[MONEY_COL].notna()].copy()
df[MONEY_COL] = pd.to_numeric(df[MONEY_COL], errors="coerce")
df = df.loc[df[MONEY_COL].notna()].copy()

print(f"Clientes totales: {df['customer_id'].nunique():,}")


# =========================
# BUILD RFM
# =========================
snapshot_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

rfm = (
    df.groupby("customer_id")
      .agg(
          last_purchase=("order_purchase_timestamp", "max"),
          frequency_orders=("order_id", "nunique"),
          monetary_value=(MONEY_COL, "sum"),
      )
      .reset_index()
)

rfm["recency_days"] = (snapshot_date - rfm["last_purchase"]).dt.days.astype(int)
rfm = rfm.drop(columns=["last_purchase"])

# winsorize monetary para evitar distorsión por outliers extremos
rfm["monetary_value"] = winsorize_series(rfm["monetary_value"], WINSOR_PCT_LOW, WINSOR_PCT_HIGH)

# sanity checks
rfm = rfm.loc[rfm["frequency_orders"] >= 1].copy()
rfm = rfm.loc[rfm["monetary_value"] >= 0].copy()

# =========================
# CLUSTERING
# =========================
X = rfm[["recency_days", "frequency_orders", "monetary_value"]].copy()

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
rfm["cluster"] = kmeans.fit_predict(X_scaled)

# =========================
# OUTPUTS
# =========================
cluster_sizes = rfm["cluster"].value_counts().sort_index()
profile = rfm.groupby("cluster")[["recency_days", "frequency_orders", "monetary_value"]].mean().sort_index()

print("\nMODEL03 — SEGMENTACIÓN CLIENTES")
print("-" * 50)

print("\nTamaño clusters:")
print(cluster_sizes)

print("\nPerfil medio clusters:")
print(profile)

# Guardados
rfm.to_csv(OUT_DIR / "customers_with_cluster.csv", index=False, encoding="utf-8")
profile.to_csv(OUT_DIR / "cluster_profile.csv", encoding="utf-8")
cluster_sizes.to_csv(OUT_DIR / "cluster_sizes.csv", encoding="utf-8")

print(f"\nOK — outputs guardados en: {OUT_DIR}")

rev_share = (
    rfm.groupby("cluster")["monetary_value"]
    .sum()
    .div(rfm["monetary_value"].sum())
    .sort_values(ascending=False)
)

print("\nRevenue share por cluster:")
print(rev_share)