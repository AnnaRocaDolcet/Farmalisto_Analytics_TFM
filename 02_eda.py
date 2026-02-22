"""
TFM Farmalisto Analytics
Script: EDA

Objetivo
- Análisis exploratorio orientado a negocio.
- Genera KPIs y visualizaciones alineadas con los 3 casos de uso: demanda, logística y cliente.

"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data_processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
KPI_DIR = OUTPUTS_DIR / "kpis"

EDA_DIR.mkdir(parents=True, exist_ok=True)
KPI_DIR.mkdir(parents=True, exist_ok=True)

FACT_FP = PROCESSED_DIR / "fact_orders.csv"

plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = False


# =========================
# HELPERS
# =========================
def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Faltan columnas necesarias en fact_orders.csv:\n"
            f"- Missing: {missing}\n"
            f"- Available: {sorted(df.columns.tolist())}"
        )


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def month_start(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").dt.to_timestamp()


def savefig(name: str) -> None:
    fp = EDA_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=300)
    plt.close()


def cap_series(s: pd.Series, cap_value: float) -> np.ndarray:
    return np.minimum(s.values, cap_value)


def rolling_mean(s: pd.Series, window: int = 3) -> pd.Series:
    return s.rolling(window=window, min_periods=1).mean()


# =========================
# LOAD
# =========================
if not FACT_FP.exists():
    raise FileNotFoundError(f"No encuentro el fichero: {FACT_FP}")

df = pd.read_csv(FACT_FP)

required = [
    "order_id",
    "customer_id",
    "order_purchase_timestamp",
    "order_status",
    "payment_value",
]
require_cols(df, required)

df["order_purchase_timestamp"] = safe_to_datetime(df["order_purchase_timestamp"])
df["order_status"] = df["order_status"].astype(str).str.lower().str.strip()
df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce")

# Opcionales (si existen)
for c in ["order_delivered_timestamp", "order_estimated_delivery_date"]:
    if c in df.columns:
        df[c] = safe_to_datetime(df[c])

# Variables logísticas derivadas si faltan
if "delivery_lead_time_days" not in df.columns and "order_delivered_timestamp" in df.columns:
    df["delivery_lead_time_days"] = (
        (df["order_delivered_timestamp"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
    )

if "delay_vs_estimated_days" not in df.columns and all(
    c in df.columns for c in ["order_delivered_timestamp", "order_estimated_delivery_date"]
):
    df["delay_vs_estimated_days"] = (
        (df["order_delivered_timestamp"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400.0
    )

if "on_time" not in df.columns and "delay_vs_estimated_days" in df.columns:
    df["on_time"] = df["delay_vs_estimated_days"] <= 0

# Limpieza mínima
df = df.loc[df["order_purchase_timestamp"].notna()].copy()
df = df.loc[df["payment_value"].notna()].copy()
df["month"] = month_start(df["order_purchase_timestamp"])


# =========================
# KPIs
# =========================
n_orders = int(df["order_id"].nunique())
n_customers = int(df["customer_id"].nunique())

gmv_total = float(df.groupby("order_id")["payment_value"].sum().sum())
ticket_mean = float(df.groupby("order_id")["payment_value"].sum().mean())

cancel_mask = df["order_status"].isin(["canceled", "cancelled"])
cancel_rate = float(cancel_mask.mean())

orders_per_customer = df.groupby("customer_id")["order_id"].nunique()
repeat_share = float((orders_per_customer >= 2).mean()) if len(orders_per_customer) else np.nan

lead_mean = np.nan
lead_p95 = np.nan
delay_mean = np.nan
late_share = np.nan
on_time_global = np.nan

if "delivery_lead_time_days" in df.columns:
    lt = pd.to_numeric(df["delivery_lead_time_days"], errors="coerce").dropna()
    lt = lt.loc[lt >= 0]
    if len(lt):
        lead_mean = float(lt.mean())
        lead_p95 = float(np.percentile(lt, 95))

if "delay_vs_estimated_days" in df.columns:
    dly = pd.to_numeric(df["delay_vs_estimated_days"], errors="coerce").dropna()
    if len(dly):
        delay_mean = float(dly.mean())
        late_share = float((dly > 0).mean())

if "on_time" in df.columns:
    on_time_global = float(pd.Series(df["on_time"]).dropna().astype(bool).mean())

# Concentración por cliente (Top 20%)
gmv_per_customer = df.groupby("customer_id")["payment_value"].sum().sort_values(ascending=False)
cut = int(np.ceil(0.2 * len(gmv_per_customer))) if len(gmv_per_customer) else 0
top20_share = (
    float(gmv_per_customer.head(cut).sum() / gmv_per_customer.sum()) if gmv_per_customer.sum() > 0 else np.nan
)

# Demanda mensual
orders_month = df.groupby("month")["order_id"].nunique().sort_index()
mom_growth = (orders_month.pct_change() * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
mom_growth_mean = float(mom_growth.mean()) if len(mom_growth) else np.nan

# Categorías 
n_categories = np.nan
top3_share = np.nan
if "dominant_category" in df.columns:
    cat_gmv = df.groupby("dominant_category")["payment_value"].sum().sort_values(ascending=False)
    n_categories = int(cat_gmv.shape[0])
    top3_share = float(cat_gmv.head(3).sum() / cat_gmv.sum()) if cat_gmv.sum() > 0 else np.nan

kpis = [
    ("n_pedidos", n_orders),
    ("n_clientes", n_customers),
    ("gmv_total_payment_value", gmv_total),
    ("ticket_medio_payment_value", ticket_mean),
    ("tasa_cancelacion", cancel_rate),
    ("share_clientes_recompra", repeat_share),
    ("on_time_rate_global", on_time_global),
    ("lead_time_medio_dias", lead_mean),
    ("lead_time_p95_dias", lead_p95),
    ("delay_medio_vs_estimado_dias", delay_mean),
    ("share_pedidos_retrasados_vs_estimado", late_share),
    ("share_gmv_top20_clientes", top20_share),
    ("crecimiento_mensual_medio_pedidos_pct", mom_growth_mean),
    ("n_categorias_dominantes", n_categories),
    ("share_gmv_top3_categorias", top3_share),
]

pd.DataFrame(kpis, columns=["kpi", "valor"]).to_csv(KPI_DIR / "kpis_eda.csv", index=False, encoding="utf-8")


# =========================
# DEMANDA
# =========================
plt.figure()
plt.plot(orders_month.index, orders_month.values)
plt.title("Evolución de pedidos por mes")
plt.xlabel("Mes")
plt.ylabel("Nº pedidos")
plt.xticks(rotation=45)
savefig("eda_01_orders_by_month")

plt.figure()
trend_3m = rolling_mean(orders_month, window=3)
plt.plot(orders_month.index, orders_month.values, label="Pedidos")
plt.plot(trend_3m.index, trend_3m.values, label="Tendencia 3m")
plt.title("Demanda mensual — tendencia (media móvil 3 meses)")
plt.xlabel("Mes")
plt.ylabel("Nº pedidos")
plt.xticks(rotation=45)
plt.legend()
savefig("eda_02_trend_3m")

plt.figure()
plt.plot((orders_month.pct_change() * 100.0).index, (orders_month.pct_change() * 100.0).values)
plt.title("Crecimiento mensual de pedidos (%)")
plt.xlabel("Mes")
plt.ylabel("Variación mensual (%)")
plt.xticks(rotation=45)
savefig("eda_03_monthly_growth_pct")

if "dominant_category" in df.columns:
    cat_gmv = df.groupby("dominant_category")["payment_value"].sum().sort_values(ascending=False)
    top10 = cat_gmv.head(10)

    plt.figure()
    plt.bar(top10.index.astype(str), top10.values)
    plt.title("Top 10 categorías por GMV (payment_value)")
    plt.xlabel("Categoría")
    plt.ylabel("GMV")
    plt.xticks(rotation=45, ha="right")
    savefig("eda_04_top_categories_gmv")

    cum = (cat_gmv / cat_gmv.sum()).cumsum()
    plt.figure()
    plt.plot(range(1, min(10, len(cum)) + 1), cum.head(10).values)
    plt.title("Pareto de GMV por categoría (concentración)")
    plt.xlabel("Ranking de categorías")
    plt.ylabel("Porcentaje acumulado de GMV")
    savefig("eda_05_pareto_categories")


# =========================
# LOGÍSTICA
# =========================
if "delivery_lead_time_days" in df.columns:
    lt = pd.to_numeric(df["delivery_lead_time_days"], errors="coerce").dropna()
    lt = lt.loc[lt >= 0]
    plt.figure()
    plt.hist(lt, bins=30)
    plt.title("Distribución del lead time (días)")
    plt.xlabel("Días desde compra hasta entrega")
    plt.ylabel("Frecuencia")
    savefig("eda_06_delivery_lead_time_hist")

if "delay_vs_estimated_days" in df.columns:
    dly = pd.to_numeric(df["delay_vs_estimated_days"], errors="coerce").dropna()
    plt.figure()
    plt.hist(dly, bins=30)
    plt.title("Delay vs fecha estimada (días)")
    plt.xlabel("Días (positivo = tarde, negativo = antes)")
    plt.ylabel("Frecuencia")
    savefig("eda_07_delay_vs_estimated_hist")

if "on_time" in df.columns:
    on_time_rate_month = df.groupby("month")["on_time"].mean().sort_index()
    plt.figure()
    plt.plot(on_time_rate_month.index, on_time_rate_month.values)
    plt.title("On-time rate por mes")
    plt.xlabel("Mes")
    plt.ylabel("On-time rate")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    savefig("eda_08_on_time_rate_by_month")


# =========================
# CLIENTE
# =========================
cap_orders = 6
opc_cap = cap_series(orders_per_customer, cap_orders)
plt.figure()
plt.hist(opc_cap, bins=np.arange(0.5, cap_orders + 1.5, 1))
plt.title("Distribución de pedidos por cliente")
plt.xlabel(f"Nº pedidos por cliente (cap visual a {cap_orders})")
plt.ylabel("Frecuencia")
savefig("eda_09_customer_orders_distribution")

cap_p99 = float(np.percentile(gmv_per_customer.values, 99)) if len(gmv_per_customer) else 0.0
gmv_cap = cap_series(gmv_per_customer, cap_p99)
plt.figure()
plt.hist(gmv_cap, bins=30)
plt.title("Distribución de GMV por cliente (payment_value)")
plt.xlabel("GMV cliente (cap p99)")
plt.ylabel("Frecuencia")
savefig("eda_10_gmv_per_customer_distribution")

plt.figure()
plt.bar(["Resto", "Top 20%"], [len(gmv_per_customer) - cut, cut])
plt.title("Concentración de clientes (Top 20% vs resto)")
plt.xlabel("Segmento")
plt.ylabel("Nº clientes")
savefig("eda_11_top20_vs_rest_customers")

print(f"Proyecto: {PROJECT_ROOT}")
print(f"Fact leído: {FACT_FP}")
print(f"KPIs guardados en: {KPI_DIR / 'kpis_eda.csv'}")
print(f"Gráficos EDA guardados en: {EDA_DIR}")