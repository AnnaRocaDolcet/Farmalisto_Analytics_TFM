"""
TFM Farmalisto Analytics
Script: Modelo 2- Logistics Delay

Objetivo:
- Predecir si un pedido llegará tarde (late=1) respecto a su fecha estimada.
- Solo usa variables disponibles ANTES de la entrega.

"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FACT_FP = PROJECT_ROOT / "data_processed" / "fact_orders.csv"

OUT_DIR = PROJECT_ROOT / "outputs" / "models" / "model02_logistics_delay"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 3
RANDOM_STATE = 42


# =========================
# HELPERS
# =========================
def month_start(ts):
    return ts.dt.to_period("M").dt.to_timestamp()

def safe_to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def compute_best_threshold_f1(y_true, y_prob):
    thresholds = np.linspace(0.05,0.95,19)
    best_t, best_f1 = 0.5,-1
    for t in thresholds:
        y_pred = (y_prob>=t).astype(int)
        p,r,f1,_ = precision_recall_fscore_support(
            y_true,y_pred,average="binary",zero_division=0
        )
        if f1>best_f1:
            best_f1=f1
            best_t=t
    return float(best_t)

def summarize(y_true,y_prob,t):
    y_pred=(y_prob>=t).astype(int)
    p,r,f1,_=precision_recall_fscore_support(
        y_true,y_pred,average="binary",zero_division=0
    )
    return {"precision":p,"recall":r,"f1":f1}

def save_plot_roc(y,p,fp):
    fpr,tpr,_=roc_curve(y,p)
    plt.figure()
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],"--")
    plt.title("ROC Curve — Logistic Delay Model")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(fp,dpi=300)
    plt.close()

def save_plot_pr(y,p,fp):
    prec,rec,_=precision_recall_curve(y,p)
    plt.figure()
    plt.plot(rec,prec)
    plt.title("Precision-Recall Curve — Logistic Delay Model")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(fp,dpi=300)
    plt.close()


# =========================
# LOAD DATA
# =========================
if not FACT_FP.exists():
    raise FileNotFoundError(f"No encuentro: {FACT_FP}")

df = pd.read_csv(FACT_FP)

required = [
    "order_id",
    "order_purchase_timestamp",
    "order_estimated_delivery_date",
    "order_delivered_timestamp",
    "order_status",
]

missing=[c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Faltan columnas: {missing}")

df["order_purchase_timestamp"]=safe_to_dt(df["order_purchase_timestamp"])
df["order_estimated_delivery_date"]=safe_to_dt(df["order_estimated_delivery_date"])
df["order_delivered_timestamp"]=safe_to_dt(df["order_delivered_timestamp"])

df["order_status"]=df["order_status"].astype(str).str.lower().str.strip()
df=df.loc[df["order_status"].eq("delivered")].copy()

df=df.loc[
    df["order_purchase_timestamp"].notna()
    & df["order_estimated_delivery_date"].notna()
    & df["order_delivered_timestamp"].notna()
].copy()

print("\nMODEL02 — LOGÍSTICA (NO LEAKAGE)")
print(f"Pedidos usados: {len(df):,}")

# TARGET
df["late"]=(df["order_delivered_timestamp"]>
            df["order_estimated_delivery_date"]).astype(int)

late_rate=float(df["late"].mean())
print(f"Tasa retraso: {late_rate:.3f}")


# =========================
# FEATURE ENGINEERING
# =========================
df["month"]=month_start(df["order_purchase_timestamp"])
df["dow"]=df["order_purchase_timestamp"].dt.dayofweek.astype("Int64")
df["hour"]=df["order_purchase_timestamp"].dt.hour.astype("Int64")

df["promised_lead_time_days"]=(
    (df["order_estimated_delivery_date"]
     -df["order_purchase_timestamp"]).dt.total_seconds()/86400
)

# tiempo aprobación
if "order_approved_at" in df.columns:
    df["order_approved_at"]=safe_to_dt(df["order_approved_at"])
    df["time_to_approve_hours"]=(
        (df["order_approved_at"]-df["order_purchase_timestamp"])
        .dt.total_seconds()/3600
    )
else:
    df["time_to_approve_hours"]=np.nan

# campos opcionales
if "items" not in df.columns:
    df["items"]=np.nan

if "payment_value" not in df.columns:
    df["payment_value"]=np.nan
df["payment_value"]=pd.to_numeric(df["payment_value"],errors="coerce")

if "dominant_category" not in df.columns:
    df["dominant_category"]="Otros"

if "customer_state" not in df.columns:
    df["customer_state"]="NA"


# =========================
# DATASET FINAL
# =========================
feature_cols_num=[
    "promised_lead_time_days",
    "time_to_approve_hours",
    "items",
    "payment_value",
    "dow",
    "hour",
]

feature_cols_cat=[
    "dominant_category",
    "customer_state",
]

df_model=df[
    ["order_id","month","late"]+feature_cols_num+feature_cols_cat
].copy()

df_model=df_model.sort_values("month").reset_index(drop=True)

X=df_model[feature_cols_num+feature_cols_cat]
y=df_model["late"].astype(int).values
months=df_model["month"]


# =========================
# PIPELINE
# =========================
numeric_transformer = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer=Pipeline([
    ("imp",SimpleImputer(strategy="most_frequent")),
    ("ohe",OneHotEncoder(handle_unknown="ignore"))
])

preprocess=ColumnTransformer([
    ("num",numeric_transformer,feature_cols_num),
    ("cat",categorical_transformer,feature_cols_cat),
])

clf = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    solver="lbfgs",
)

model=Pipeline([
    ("prep",preprocess),
    ("clf",clf)
])


# =========================
# BASELINE
# =========================
baseline=np.zeros_like(y)
p,r,f,_=precision_recall_fscore_support(
    y,baseline,average="binary",zero_division=0
)

print("\nBaseline:")
print(f"Precision={p:.3f} Recall={r:.3f} F1={f:.3f}")


# =========================
# CV TEMPORAL
# =========================
tscv=TimeSeriesSplit(n_splits=N_SPLITS)

oof_prob=np.zeros(len(df_model))
oof_fold=np.full(len(df_model),-1)

rows=[]

for fold,(tr,te) in enumerate(tscv.split(X)):

    model.fit(X.iloc[tr],y[tr])
    prob=model.predict_proba(X.iloc[te])[:,1]

    oof_prob[te]=prob
    oof_fold[te]=fold

    roc=roc_auc_score(y[te],prob) if len(np.unique(y[te]))>1 else np.nan
    ap=average_precision_score(y[te],prob)

    s05=summarize(y[te],prob,0.5)

    rows.append({
        "fold":fold,
        "roc_auc":roc,
        "avg_precision":ap,
        "precision@0.5":s05["precision"],
        "recall@0.5":s05["recall"],
        "f1@0.5":s05["f1"],
    })

fold_df=pd.DataFrame(rows)

print("\nCV medias:")
print(fold_df.mean(numeric_only=True))


# =========================
# OOF GLOBAL
# =========================
mask=oof_fold>=0
y_oof=y[mask]
p_oof=oof_prob[mask]

roc_oof=roc_auc_score(y_oof,p_oof)
ap_oof=average_precision_score(y_oof,p_oof)

thr=compute_best_threshold_f1(y_oof,p_oof)
s05=summarize(y_oof,p_oof,0.5)
sb=summarize(y_oof,p_oof,thr)

print("\nOOF:")
print(f"ROC-AUC={roc_oof:.3f}")
print(f"PR-AUC={ap_oof:.3f}")
print(f"Best threshold={thr:.2f}")

print("\nConfusion 0.5")
print(confusion_matrix(y_oof,(p_oof>=0.5)))

print(f"\nConfusion {thr:.2f}")
print(confusion_matrix(y_oof,(p_oof>=thr)))


# =========================
# SAVE OUTPUTS
# =========================
metrics=pd.DataFrame([{
    "orders":len(df_model),
    "late_rate":late_rate,
    "roc_auc":roc_oof,
    "avg_precision":ap_oof,
    "best_thr":thr,
}])

metrics.to_csv(OUT_DIR/"metrics.csv",index=False)
fold_df.to_csv(OUT_DIR/"cv_folds.csv",index=False)

pred=df_model[["order_id","month"]].copy()
pred["prob_late"]=oof_prob
pred.to_csv(OUT_DIR/"predictions_oof.csv",index=False)

save_plot_roc(y_oof,p_oof,PLOTS_DIR/"roc.png")
save_plot_pr(y_oof,p_oof,PLOTS_DIR/"pr.png")

print("\nOK outputs guardados en:")
print(OUT_DIR)