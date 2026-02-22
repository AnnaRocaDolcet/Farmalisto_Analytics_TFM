"""
TFM Farmalisto Analytics
Script: Modelo 2 - Demand Forecasting

Objetivo:
•⁠  ⁠Estimar la demanda mensual (número de pedidos) a partir del histórico transaccional.
•⁠  ⁠Evaluar contra baselines sólidos (Naive-1, Seasonal-Naive-12, MA-3) y un modelo ML simple (Ridge).
•⁠  ⁠Seleccionar el “winner” por categoría (mínimo WAPE; desempate por MAE) y guardar artefactos.

"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


# =========================
# CONFIGURACIÓN
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Farmalisto_TFM
FACT_FP = PROJECT_ROOT / "data_processed" / "fact_orders.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "model_01_forecast_demand"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

WINNERS_DIR = OUT_DIR / "winners"
WINNERS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_CATEGORIES = 6
TEST_HORIZON_MONTHS = 3
MIN_MONTHS_REQUIRED = 12  # umbral mínimo para considerar la serie utilizable


# =========================
# ESTRUCTURAS PARA GUARDAR MODELOS
# =========================
trained_models: dict[tuple[str, str], object] = {}       # solo modelos entrenables (Ridge)
baseline_artifacts: dict[tuple[str, str], dict] = {}     # “modelos” baseline como parámetros reproducibles


# =========================
# HELPERS
# =========================
def month_start(ts: pd.Series) -> pd.Series:
    return ts.dt.to_period("M").dt.to_timestamp()


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def safe_clip_nonnegative(a: np.ndarray) -> np.ndarray:
    return np.maximum(a, 0)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features temporales simples, robustas y defendibles:
    - lag1, lag2, lag3
    - roll3: media móvil 3 (hasta t-1)
    - month_num: mes del año (1..12)
    """
    df = df.copy()
    df["lag1"] = df["y"].shift(1)
    df["lag2"] = df["y"].shift(2)
    df["lag3"] = df["y"].shift(3)
    df["roll3"] = df["y"].shift(1).rolling(3, min_periods=1).mean()
    df["month_num"] = df["month"].dt.month
    return df


def temporal_train_test_split(series_df: pd.DataFrame, test_h: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal: últimos test_h meses como test.
    series_df: columnas [month, y]
    """
    series_df = series_df.sort_values("month").reset_index(drop=True)
    if len(series_df) <= test_h + 3:
        return series_df.iloc[:0], series_df
    train = series_df.iloc[:-test_h].copy()
    test = series_df.iloc[-test_h:].copy()
    return train, test


def forecast_baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Baselines defendibles:
    - baseline_naive_lag1: último valor observado
    - baseline_ma3: media de los últimos 3 meses de train
    - baseline_seasonal_naive12: valor del mismo mes del año anterior (si existe), fallback a lag1
    """
    y_train = train["y"].values.astype(float)
    preds: dict[str, np.ndarray] = {}

    # Naive lag-1
    last = float(y_train[-1]) if len(y_train) else 0.0
    preds["baseline_naive_lag1"] = np.full(len(test), last, dtype=float)

    # MA-3
    if len(y_train) >= 3:
        ma3 = float(np.mean(y_train[-3:]))
    elif len(y_train) > 0:
        ma3 = float(np.mean(y_train))
    else:
        ma3 = 0.0
    preds["baseline_ma3"] = np.full(len(test), ma3, dtype=float)

    # Seasonal Naive-12
    seasonal = []
    train_map = dict(zip(train["month"].dt.to_period("M"), train["y"].astype(float)))
    for m in test["month"].dt.to_period("M"):
        m_lag12 = (m - 12)
        seasonal.append(float(train_map.get(m_lag12, last)))
    preds["baseline_seasonal_naive12"] = np.array(seasonal, dtype=float)

    return preds


def forecast_ridge(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, object | None]:
    """
    Ridge regression con features temporales simples.
    Devuelve:
    - predicciones (test)
    - modelo entrenado (o None si cae a baseline)
    """
    train_f = add_time_features(train)

    # Para construir lags/rolling en test, concatenamos cola de train con test
    tmp = pd.concat([train.tail(3), test], ignore_index=True)
    test_f = add_time_features(tmp).iloc[-len(test):].copy()

    feats = ["lag1", "lag2", "lag3", "roll3", "month_num"]
    train_f = train_f.dropna(subset=feats + ["y"]).copy()

    # Si no hay suficiente data para entrenar
    if len(train_f) < 6:
        last = float(train["y"].iloc[-1]) if len(train) else 0.0
        return np.full(len(test), last, dtype=float), None

    X_train = train_f[feats].values
    y_train = train_f["y"].values.astype(float)

    X_test = test_f[feats].ffill().bfill().values

    model = Ridge(alpha=1.0, random_state=0)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    pred = safe_clip_nonnegative(pred)
    return pred, model


# =========================
# CARGA Y PREPARACIÓN
# =========================
if not FACT_FP.exists():
    raise FileNotFoundError(f"No encuentro: {FACT_FP}")

df = pd.read_csv(FACT_FP)

required = ["order_id", "order_purchase_timestamp", "dominant_category"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(
        f"Faltan columnas en fact_orders.csv: {missing}\n"
        f"Disponibles: {sorted(df.columns.tolist())}"
    )

df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
df = df.loc[df["order_purchase_timestamp"].notna()].copy()
df["month"] = month_start(df["order_purchase_timestamp"])

# Demanda mensual por categoría (nº pedidos únicos)
monthly = (
    df.groupby(["month", "dominant_category"])["order_id"]
      .nunique()
      .reset_index(name="y")
      .sort_values(["dominant_category", "month"])
)

# Top N categorías por volumen total (más estable con histórico corto)
cat_volume = monthly.groupby("dominant_category")["y"].sum().sort_values(ascending=False)
top_cats = cat_volume.head(TOP_N_CATEGORIES).index.tolist()
monthly = monthly.loc[monthly["dominant_category"].isin(top_cats)].copy()

# Forzar continuidad mensual por categoría (meses faltantes => 0)
all_months = pd.date_range(monthly["month"].min(), monthly["month"].max(), freq="MS")
series_list = []
for cat in top_cats:
    s = (
        monthly.loc[monthly["dominant_category"] == cat, ["month", "y"]]
        .set_index("month")
        .reindex(all_months)
    )
    s["y"] = s["y"].fillna(0)
    s = s.reset_index().rename(columns={"index": "month"})
    s["dominant_category"] = cat
    series_list.append(s)

monthly_full = pd.concat(series_list, ignore_index=True)

n_months = monthly_full["month"].nunique()
print(f"Meses únicos disponibles: {n_months} | Rango: {monthly_full['month'].min().date()} -> {monthly_full['month'].max().date()}")
print(f"Top categorías seleccionadas (por volumen): {top_cats}")


# =========================
# TRAIN/TEST + FORECAST
# =========================
results: list[dict] = []
pred_rows: list[dict] = []

for cat in top_cats:
    s = monthly_full.loc[monthly_full["dominant_category"] == cat, ["month", "y"]].copy().sort_values("month")

    if s["month"].nunique() < MIN_MONTHS_REQUIRED:
        continue

    train, test = temporal_train_test_split(s, TEST_HORIZON_MONTHS)
    if len(train) == 0 or len(test) == 0:
        continue

    y_true = test["y"].values.astype(float)

    # Baselines
    base_preds = forecast_baselines(train, test)

    # Guardar artefactos baseline (parámetros reproducibles)
    y_train = train["y"].values.astype(float)
    last = float(y_train[-1]) if len(y_train) else 0.0
    ma3 = float(np.mean(y_train[-3:])) if len(y_train) >= 3 else float(np.mean(y_train)) if len(y_train) else 0.0

    baseline_artifacts[(cat, "baseline_naive_lag1")] = {"type": "naive_lag1", "last_value": last}
    baseline_artifacts[(cat, "baseline_ma3")] = {"type": "ma3", "ma3_value": ma3}
    baseline_artifacts[(cat, "baseline_seasonal_naive12")] = {"type": "seasonal_naive12", "fallback_last": last}

    for name, y_pred in base_preds.items():
        mae = float(mean_absolute_error(y_true, y_pred))
        results.append({
            "category": cat,
            "model": name,
            "mae": mae,
            "wape": wape(y_true, y_pred),
            "test_months": len(test),
            "train_months": len(train),
        })
        for m, yt, yp in zip(test["month"], y_true, y_pred):
            pred_rows.append({
                "category": cat,
                "model": name,
                "month": m,
                "y_true": float(yt),
                "y_pred": float(yp),
            })

    # Ridge
    ridge_pred, ridge_model = forecast_ridge(train, test)
    if ridge_model is not None:
        trained_models[(cat, "ridge_time_features")] = ridge_model

    mae = float(mean_absolute_error(y_true, ridge_pred))
    results.append({
        "category": cat,
        "model": "ridge_time_features",
        "mae": mae,
        "wape": wape(y_true, ridge_pred),
        "test_months": len(test),
        "train_months": len(train),
    })
    for m, yt, yp in zip(test["month"], y_true, ridge_pred):
        pred_rows.append({
            "category": cat,
            "model": "ridge_time_features",
            "month": m,
            "y_true": float(yt),
            "y_pred": float(yp),
        })

    # Plot backtest (storytelling)
    last_k = 6 + TEST_HORIZON_MONTHS
    tail = s.tail(last_k).copy()

    plt.figure()
    plt.plot(tail["month"], tail["y"], label="Real")
    plt.plot(test["month"], ridge_pred, label="Pred Ridge (test)")
    plt.title(f"Backtest demanda mensual — {cat} (últimos {last_k} meses)")
    plt.xlabel("Mes")
    plt.ylabel("Pedidos")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    safe_cat = str(cat).replace("/", "_").replace(" ", "_")
    plt.savefig(PLOTS_DIR / f"forecast_backtest_{safe_cat}.png", dpi=300)
    plt.close()


# =========================
# OUTPUTS (MÉTRICAS Y PREDICCIONES)
# =========================
metrics_df = pd.DataFrame(results)
pred_df = pd.DataFrame(pred_rows)

metrics_fp = OUT_DIR / "forecast_metrics.csv"
pred_fp = OUT_DIR / "forecast_predictions_test.csv"

metrics_df.to_csv(metrics_fp, index=False, encoding="utf-8")
pred_df.to_csv(pred_fp, index=False, encoding="utf-8")

# Resumen en consola: mejor modelo por categoría (WAPE; desempate por MAE)
if not metrics_df.empty:
    best = (
        metrics_df.sort_values(["category", "wape", "mae"], ascending=[True, True, True])
                  .groupby("category", as_index=False)
                  .first()
    )

    print("\nFORECAST RESULTS — DEMANDA (mensual x categoría) — mejores por WAPE")
    print("-" * 70)
    for _, r in best.iterrows():
        print(f"{r['category']:<30} | {r['model']:<25} | MAE={r['mae']:.3f} | WAPE={r['wape']:.3f}")
    print("-" * 70)
else:
    print("No se generaron métricas (histórico insuficiente o filtros demasiado estrictos).")

print(f"\nOK — métricas guardadas en: {metrics_fp}")
print(f"OK — predicciones guardadas en: {pred_fp}")
print(f"OK — plots guardados en: {PLOTS_DIR}")


# =========================
# SAVE WINNERS (MODELOS + RESUMEN EJECUTIVO)
# =========================
if not metrics_df.empty:
    winners = (
        metrics_df.sort_values(["category", "wape", "mae"], ascending=[True, True, True])
        .groupby("category", as_index=False)
        .first()
        .rename(columns={"model": "winner_model", "mae": "winner_mae", "wape": "winner_wape"})
    )

    rows: list[dict] = []
    for _, r in winners.iterrows():
        cat = r["category"]
        model_name = r["winner_model"]
        safe_cat = str(cat).replace("/", "_").replace(" ", "_")

        meta = {
            "category": cat,
            "winner_model": model_name,
            "mae": float(r["winner_mae"]),
            "wape": float(r["winner_wape"]),
        }

        # Winner = Ridge => joblib
        if (cat, model_name) in trained_models:
            model_obj = trained_models[(cat, model_name)]
            model_fp = WINNERS_DIR / f"model_{safe_cat}.joblib"
            joblib.dump(model_obj, model_fp)

        # Winner = baseline => json con parámetros
        elif (cat, model_name) in baseline_artifacts:
            baseline_fp = WINNERS_DIR / f"model_{safe_cat}.json"
            with open(baseline_fp, "w", encoding="utf-8") as f:
                json.dump(baseline_artifacts[(cat, model_name)], f, ensure_ascii=False, indent=2)

        rows.append(meta)

    winners_summary = pd.DataFrame(rows).sort_values(["wape", "mae"])
    winners_fp = OUT_DIR / "forecast_winners_summary.csv"
    winners_summary.to_csv(winners_fp, index=False, encoding="utf-8")

    print(f"OK — winners guardados en: {WINNERS_DIR}")
    print(f"OK — resumen winners guardado en: {winners_fp}")