"""
TFM Farmalisto Analytics
Script: Data mining

Objetivo:
- Minería de patrones accionables
  - Market Basket/afinidad: co-ocurrencias por cliente o secuencias.
  - Cohorts/retención por mes: minería temporal de comportamiento.
  - ABC/Pareto formal por categoría: clasificación operativa.

"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACT_FP = PROJECT_ROOT / "data_processed" / "fact_orders.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "data_mining"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def safe_to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def month_start(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").dt.to_timestamp()


def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Faltan columnas necesarias en fact_orders.csv: {missing}\n"
            f"Disponibles: {sorted(df.columns.tolist())}"
        )


def build_market_basket_rules(
    df: pd.DataFrame,
    min_support: float = 0.005,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Market Basket 'light' a nivel de categoría dominante por cliente.
    Genera reglas X -> Y con support, confidence y lift.

    min_support: proporción mínima de clientes que deben contener el par (X,Y).
    """
    base = df[["customer_id", "dominant_category"]].dropna().copy()
    base["dominant_category"] = base["dominant_category"].astype(str).str.strip()

    cust_cat = (
        base.drop_duplicates(["customer_id", "dominant_category"])
        .assign(flag=1)
        .pivot_table(index="customer_id", columns="dominant_category", values="flag", fill_value=0)
    )

    if cust_cat.shape[1] < 2:
        return pd.DataFrame(columns=[
            "antecedent", "consequent", "support", "confidence", "lift",
            "support_antecedent", "support_consequent",
            "n_customers_total", "n_customers_both"
        ])

    M = cust_cat.to_numpy(dtype=np.int32)
    cats = cust_cat.columns.to_list()
    n_customers = M.shape[0]

    support_single = M.mean(axis=0)

    cooc_counts = (M.T @ M).astype(np.int64)
    np.fill_diagonal(cooc_counts, 0)

    rows = []
    for i in range(len(cats)):
        for j in range(len(cats)):
            if i == j:
                continue

            n_both = int(cooc_counts[i, j])
            support_xy = n_both / n_customers

            if support_xy < min_support:
                continue

            support_x = float(support_single[i])
            support_y = float(support_single[j])

            if support_x <= 0 or support_y <= 0:
                continue

            confidence = support_xy / support_x
            lift = confidence / support_y

            rows.append({
                "antecedent": cats[i],
                "consequent": cats[j],
                "support": float(support_xy),
                "confidence": float(confidence),
                "lift": float(lift),
                "support_antecedent": float(support_x),
                "support_consequent": float(support_y),
                "n_customers_total": int(n_customers),
                "n_customers_both": int(n_both),
            })

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules

    rules = rules.sort_values(["lift", "confidence", "support"], ascending=[False, False, False]).head(top_n)
    return rules.reset_index(drop=True)


def build_cohort_retention(df: pd.DataFrame, max_months: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cohorts por mes de primera compra.
    Retención: % clientes activos en mes +k respecto al tamaño de cohorte.
    Devuelve:
      - long_df: cohort_month, period_index, customers, cohort_size, retention_rate
      - matrix_df: tabla cohort_month x period_index (retention_rate)
    """
    base = df[["customer_id", "order_purchase_timestamp"]].dropna().copy()
    base["order_purchase_timestamp"] = safe_to_dt(base["order_purchase_timestamp"])
    base = base.loc[base["order_purchase_timestamp"].notna()].copy()

    base["order_month"] = month_start(base["order_purchase_timestamp"])

    first_month = base.groupby("customer_id", as_index=False)["order_month"].min()
    first_month = first_month.rename(columns={"order_month": "cohort_month"})

    base = base.merge(first_month, on="customer_id", how="left")

    base["period_index"] = (
        (base["order_month"].dt.year - base["cohort_month"].dt.year) * 12
        + (base["order_month"].dt.month - base["cohort_month"].dt.month)
    )

    base = base.loc[(base["period_index"] >= 0) & (base["period_index"] <= max_months)].copy()

    cohort_sizes = base.loc[base["period_index"] == 0].groupby("cohort_month")["customer_id"].nunique()
    active = base.groupby(["cohort_month", "period_index"])["customer_id"].nunique()

    long_df = active.reset_index(name="customers")
    long_df["cohort_size"] = long_df["cohort_month"].map(cohort_sizes).astype(float)
    long_df["retention_rate"] = long_df["customers"] / long_df["cohort_size"]
    long_df = long_df.sort_values(["cohort_month", "period_index"]).reset_index(drop=True)

    matrix_df = long_df.pivot(index="cohort_month", columns="period_index", values="retention_rate").sort_index()
    return long_df, matrix_df


def build_abc_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasificación ABC por GMV (payment_value) a nivel de categoría dominante.
      - A: hasta 80% acumulado
      - B: 80–95%
      - C: 95–100%
    """
    base = df[["dominant_category", "payment_value"]].copy()
    base["dominant_category"] = base["dominant_category"].fillna("Otros").astype(str).str.strip()
    base["payment_value"] = pd.to_numeric(base["payment_value"], errors="coerce")

    base = base.loc[base["payment_value"].notna() & (base["payment_value"] >= 0)].copy()

    gmv = base.groupby("dominant_category", as_index=False)["payment_value"].sum()
    gmv = gmv.rename(columns={"payment_value": "gmv"})
    gmv = gmv.sort_values("gmv", ascending=False).reset_index(drop=True)

    total = float(gmv["gmv"].sum())
    if total <= 0:
        gmv["gmv_share"] = np.nan
        gmv["gmv_share_cum"] = np.nan
        gmv["abc_class"] = "NA"
        return gmv

    gmv["gmv_share"] = gmv["gmv"] / total
    gmv["gmv_share_cum"] = gmv["gmv_share"].cumsum()

    def abc_label(x: float) -> str:
        if x <= 0.80:
            return "A"
        if x <= 0.95:
            return "B"
        return "C"

    gmv["abc_class"] = gmv["gmv_share_cum"].apply(abc_label)
    return gmv


def plot_cohort_heatmap(matrix_df: pd.DataFrame, fp: Path) -> None:
    if matrix_df.empty:
        return

    mat = matrix_df.to_numpy(dtype=float)
    plt.figure(figsize=(12, 6))
    plt.imshow(mat, aspect="auto")

    plt.title("Cohort Retention Heatmap (Monthly)")
    plt.xlabel("Meses desde primera compra (period_index)")
    plt.ylabel("Cohorte (mes primera compra)")

    plt.xticks(ticks=np.arange(matrix_df.shape[1]), labels=[str(c) for c in matrix_df.columns])
    plt.yticks(ticks=np.arange(matrix_df.shape[0]), labels=[d.strftime("%Y-%m") for d in matrix_df.index])

    cbar = plt.colorbar()
    cbar.set_label("Retention rate")

    plt.tight_layout()
    plt.savefig(fp, dpi=300)
    plt.close()


def main() -> None:
    if not FACT_FP.exists():
        raise FileNotFoundError(f"No encuentro: {FACT_FP}")

    df = pd.read_csv(FACT_FP)

    require_cols(df, [
        "order_id",
        "customer_id",
        "order_purchase_timestamp",
        "dominant_category",
        "payment_value",
        "order_status",
    ])

    df["order_purchase_timestamp"] = safe_to_dt(df["order_purchase_timestamp"])
    df["order_status"] = df["order_status"].astype(str).str.lower().str.strip()

    df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce")

    df = df.loc[df["order_purchase_timestamp"].notna()].copy()
    df = df.loc[df["payment_value"].notna()].copy()

    print("\nDATA MINING — Farmalisto")
    print("-" * 60)
    print(f"Pedidos en fact (válidos): {len(df):,}")
    print(f"Clientes únicos: {df['customer_id'].nunique():,}")
    print(f"GMV total (payment_value): {df['payment_value'].sum():,.2f}")

    rules = build_market_basket_rules(df, min_support=0.005, top_n=30)
    rules_fp = OUT_DIR / "market_basket_rules.csv"
    rules.to_csv(rules_fp, index=False, encoding="utf-8")

    cohort_long, cohort_matrix = build_cohort_retention(df, max_months=12)
    cohort_long_fp = OUT_DIR / "cohort_retention_long.csv"
    cohort_matrix_fp = OUT_DIR / "cohort_retention_matrix.csv"
    cohort_long.to_csv(cohort_long_fp, index=False, encoding="utf-8")
    cohort_matrix.to_csv(cohort_matrix_fp, index=True, encoding="utf-8")

    heatmap_fp = PLOTS_DIR / "cohort_retention_heatmap.png"
    plot_cohort_heatmap(cohort_matrix, heatmap_fp)

    abc_df = build_abc_categories(df)
    abc_fp = OUT_DIR / "abc_categories.csv"
    abc_df.to_csv(abc_fp, index=False, encoding="utf-8")

    print("\nOK — outputs guardados en:")
    print(f"- {rules_fp}")
    print(f"- {cohort_long_fp}")
    print(f"- {cohort_matrix_fp}")
    print(f"- {abc_fp}")
    print(f"- {heatmap_fp}")


if __name__ == "__main__":
    main()