"""
TFM Farmalisto Analytics
Script: Build dataset

Objetivo:
- Construcción del dataset mínimo viable a partir de datos Kaggle (train), con adaptación a:
    - Dominio e-pharma.
    - Ajuste temporal.
    - Simulación controlada.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    target_max_date: pd.Timestamp = pd.Timestamp("2026-02-01")
    random_seed: int = 42

    # Simulación controlada
    repeat_rate: float = 0.18
    cancel_rate: float = 0.07

    epharma_categories: tuple[str, ...] = (
        "Medicamentos con receta",
        "OTC & cuidado personal",
        "Suplementos",
        "Higiene y bienestar",
        "Pediatría",
        "Primeros auxilios y cuidado",
        "Dispositivos médicos hogar",
        "Accesorios sanitarios",
        "Monitorización y diagnóstico",
        "Salud digital (servicios)",
    )


CFG = Config()


# =========================
# HELPERS
# =========================
def project_root_from_this_file() -> Path:
    """Asume que el script vive en /Farmalisto_TFM/notebooks/."""
    here = Path(__file__).resolve()
    return here.parents[1]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas a snake_case minúsculas."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("__", "_", regex=False)
        .str.lower()
    )
    return df


def find_required_files(train_dir: Path) -> dict[str, Path]:
    """
    Localiza los CSV requeridos en la carpeta train.
    Requiere (lógicos): orders, orderitems, customers, payments, products.
    """
    if not train_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {train_dir}")

    csvs = list(train_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No se han encontrado CSVs en: {train_dir}")

    by_key = {fp.stem.lower(): fp for fp in csvs}

    candidates = {
        "orders": ["df_orders", "df_orders_dataset", "orders"],
        "orderitems": ["df_orderitems", "df_order_items", "orderitems", "order_items"],
        "customers": ["df_customers", "customers"],
        "payments": ["df_payments", "payments"],
        "products": ["df_products", "products"],
    }

    def pick(logical: str) -> Path:
        for cand in candidates[logical]:
            if cand in by_key:
                return by_key[cand]
        for stem, fp in by_key.items():
            for cand in candidates[logical]:
                if cand.replace("_", "") in stem.replace("_", ""):
                    return fp
        raise FileNotFoundError(
            f"No se ha podido localizar el fichero para '{logical}' en {train_dir}.\n"
            f"CSVs disponibles: {[p.name for p in csvs]}"
        )

    return {
        "orders": pick("orders"),
        "order_items": pick("orderitems"),
        "customers": pick("customers"),
        "payments": pick("payments"),
        "products": pick("products"),
    }


def parse_datetimes(orders: pd.DataFrame) -> pd.DataFrame:
    """Convierte timestamps a datetime cuando existan."""
    orders = orders.copy()
    dt_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_timestamp",
        "order_estimated_delivery_date",
    ]
    for c in dt_cols:
        if c in orders.columns:
            orders[c] = pd.to_datetime(orders[c], errors="coerce")
    return orders


def shift_dates_to_target(orders: pd.DataFrame, target_max_date: pd.Timestamp) -> pd.DataFrame:
    """
    Desplaza fechas para que el máximo 'order_delivered_timestamp' coincida con target_max_date,
    preservando diferencias relativas.
    """
    orders = orders.copy()
    if "order_delivered_timestamp" not in orders.columns:
        return orders

    max_orig = orders["order_delivered_timestamp"].max()
    if pd.isna(max_orig):
        return orders

    delta = target_max_date - max_orig
    for c in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_timestamp",
        "order_estimated_delivery_date",
    ]:
        if c in orders.columns:
            orders[c] = orders[c] + delta

    print(f"Fechas ajustadas: max original={max_orig.date()} -> max nuevo={target_max_date.date()}")
    return orders


def deterministic_category_map(series: pd.Series, epharma_categories: tuple[str, ...]) -> pd.Series:
    """Mapeo determinista de categorías a catálogo e-pharma."""
    def map_one(x: str) -> str:
        if pd.isna(x) or str(x).strip() == "":
            return "Otros"
        s = str(x).strip().lower()
        h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        return epharma_categories[h % len(epharma_categories)]

    return series.apply(map_one)


def simulate_repeat_customers(orders: pd.DataFrame, repeat_rate: float, seed: int) -> pd.DataFrame:
    """Simula recurrencia reasignando el customer_id en una fracción de pedidos."""
    orders = orders.copy()
    if "customer_id" not in orders.columns or len(orders) == 0:
        return orders

    rng = np.random.default_rng(seed)
    mask = rng.random(len(orders)) < repeat_rate

    unique_customers = orders["customer_id"].dropna().unique()
    if len(unique_customers) == 0:
        return orders

    pool_size = max(1, int(0.25 * len(unique_customers)))
    pool = rng.choice(unique_customers, size=pool_size, replace=False)
    orders.loc[mask, "customer_id"] = rng.choice(pool, size=int(mask.sum()), replace=True)
    return orders


def simulate_cancellations(orders: pd.DataFrame, cancel_rate: float, seed: int) -> pd.DataFrame:
    """Simula cancelaciones marcando una fracción de pedidos como canceled."""
    orders = orders.copy()
    if "order_status" not in orders.columns:
        orders["order_status"] = "delivered"

    if len(orders) == 0:
        return orders

    rng = np.random.default_rng(seed + 1)
    mask = rng.random(len(orders)) < cancel_rate

    orders.loc[mask, "order_status"] = "canceled"
    if "order_delivered_timestamp" in orders.columns:
        orders.loc[mask, "order_delivered_timestamp"] = pd.NaT

    return orders


def build_fact_orders(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    payments: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye tabla fact a nivel pedido (order_id).
    Variables económicas: payment_value (suma de pagos por pedido).
    Categoría dominante: categoría modal del pedido (por nº de items).
    Variables logísticas derivadas: lead time, delay vs estimado, on_time.
    """
    fact = orders.copy()

    # Items por pedido
    oi = order_items.copy()
    if "order_id" in oi.columns:
        if "order_item_id" in oi.columns:
            items_agg = oi.groupby("order_id", as_index=False).agg(items=("order_item_id", "count"))
        else:
            items_agg = oi.groupby("order_id", as_index=False).agg(items=("product_id", "count"))
        fact = fact.merge(items_agg, on="order_id", how="left")
    else:
        fact["items"] = np.nan

    # Payments por pedido (variable económica estándar)
    pay = payments.copy()
    if "order_id" in pay.columns and "payment_value" in pay.columns:
        pay["payment_value"] = pd.to_numeric(pay["payment_value"], errors="coerce")
        pay_agg = pay.groupby("order_id", as_index=False)["payment_value"].sum()
        fact = fact.merge(pay_agg, on="order_id", how="left")
    else:
        fact["payment_value"] = np.nan

    # Categoría dominante por pedido (modal por nº de items)
    p = products.copy()
    if "product_category_name" in p.columns:
        p = p.rename(columns={"product_category_name": "category"})
    else:
        p["category"] = "Otros"

    if all(c in oi.columns for c in ["product_id", "order_id"]):
        oi_cat = oi.merge(p[["product_id", "category"]], on="product_id", how="left")
        cat_mode = (
            oi_cat.groupby(["order_id", "category"], as_index=False)
            .size()
            .sort_values(["order_id", "size"], ascending=[True, False])
            .drop_duplicates("order_id")
            .rename(columns={"category": "dominant_category"})
            [["order_id", "dominant_category"]]
        )
        fact = fact.merge(cat_mode, on="order_id", how="left")
    else:
        fact["dominant_category"] = "Otros"

    # Geografía cliente
    c = customers.copy()
    geo_cols = [col for col in ["customer_city", "customer_state", "customer_zip_code_prefix"] if col in c.columns]
    if "customer_id" in fact.columns and "customer_id" in c.columns and geo_cols:
        fact = fact.merge(c[["customer_id"] + geo_cols], on="customer_id", how="left")

    # Variables logísticas
    if all(col in fact.columns for col in ["order_purchase_timestamp", "order_delivered_timestamp"]):
        fact["delivery_lead_time_days"] = (
            (fact["order_delivered_timestamp"] - fact["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
        )

    if all(col in fact.columns for col in ["order_estimated_delivery_date", "order_delivered_timestamp"]):
        fact["delay_vs_estimated_days"] = (
            (fact["order_delivered_timestamp"] - fact["order_estimated_delivery_date"]).dt.total_seconds() / 86400.0
        )

    if "delay_vs_estimated_days" in fact.columns:
        fact["on_time"] = fact["delay_vs_estimated_days"].le(0)

    return fact


# =========================
# MAIN
# =========================
def main() -> None:
    np.random.seed(CFG.random_seed)

    project_root = project_root_from_this_file()
    raw_train = project_root / "data_raw" / "kaggle_ecommerce" / "train"
    processed = project_root / "data_processed"
    processed.mkdir(parents=True, exist_ok=True)

    print(f"Proyecto: {project_root}")
    print(f"Input train: {raw_train}")
    print(f"Output processed: {processed}")

    files = find_required_files(raw_train)

    orders = normalize_columns(pd.read_csv(files["orders"]))
    order_items = normalize_columns(pd.read_csv(files["order_items"]))
    customers = normalize_columns(pd.read_csv(files["customers"]))
    payments = normalize_columns(pd.read_csv(files["payments"]))
    products = normalize_columns(pd.read_csv(files["products"]))

    orders = parse_datetimes(orders)
    orders = shift_dates_to_target(orders, CFG.target_max_date)

    # Adaptación categorías
    if "product_category_name" in products.columns:
        products["product_category_name"] = deterministic_category_map(
            products["product_category_name"], CFG.epharma_categories
        )
        print("Categorías adaptadas a e-pharma (mapeo determinista).")
    else:
        products["product_category_name"] = "Otros"
        print("products sin 'product_category_name': se asigna 'Otros'.")

    # Simulación controlada
    orders = simulate_repeat_customers(orders, CFG.repeat_rate, CFG.random_seed)
    orders = simulate_cancellations(orders, CFG.cancel_rate, CFG.random_seed)

    # Coherencia cancelados
    if "order_status" in orders.columns and "order_delivered_timestamp" in orders.columns:
        canceled = orders["order_status"].astype(str).str.lower().eq("canceled")
        orders.loc[canceled, "order_delivered_timestamp"] = pd.NaT

    # Guardado tablas procesadas
    (processed / "orders_processed.csv").write_text("", encoding="utf-8")  # placeholder for atomic save check
    orders.to_csv(processed / "orders_processed.csv", index=False)
    order_items.to_csv(processed / "order_items_processed.csv", index=False)
    customers.to_csv(processed / "customers_processed.csv", index=False)
    payments.to_csv(processed / "payments_processed.csv", index=False)
    products.to_csv(processed / "products_processed.csv", index=False)

    # Fact table
    fact_orders = build_fact_orders(orders, order_items, payments, customers, products)
    out_fact = processed / "fact_orders.csv"
    fact_orders.to_csv(out_fact, index=False)

    # Sanity checks
    total_orders = int(orders["order_id"].nunique()) if "order_id" in orders.columns else len(orders)

    cancel_rate = np.nan
    if "order_status" in orders.columns:
        cancel_rate = float(orders["order_status"].astype(str).str.lower().eq("canceled").mean())

    repeat_share = np.nan
    if "customer_id" in orders.columns and "order_id" in orders.columns:
        opc = orders.groupby("customer_id")["order_id"].nunique()
        repeat_share = float((opc >= 2).mean()) if len(opc) else np.nan

    print("OK — dataset procesado guardado en:", processed)
    print(f"Sanity — pedidos={total_orders:,} | cancel_rate(sim)={cancel_rate:.3f} | repeat_customers_share(sim)={repeat_share:.3f}")
    print("OK — fact_orders guardado en:", out_fact)


if __name__ == "__main__":
    main()