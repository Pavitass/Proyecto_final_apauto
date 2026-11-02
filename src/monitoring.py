"""Monitoreo mensual del modelo de riesgo (punto 3.9 de la propuesta)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from features import TARGET

PSI_MODERATE, PSI_HIGH = 0.10, 0.25
RECALL_DROP = 0.05


def psi(ref, new, bins=10, eps=1e-4):
    """Population Stability Index. Quantile bins from the reference; missing values get their own bin."""
    ref, new = pd.Series(ref, dtype=float), pd.Series(new, dtype=float)
    r, n = ref.dropna(), new.dropna()
    if r.nunique() <= bins:
        cats = sorted(set(r.unique()) | set(n.unique()))
        rc, nc = r.value_counts().reindex(cats, fill_value=0), n.value_counts().reindex(cats, fill_value=0)
    else:
        edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        rc, nc = pd.cut(r, edges).value_counts(sort=False), pd.cut(n, edges).value_counts(sort=False)
    rc = np.append(rc.values, ref.isna().sum()).astype(float)
    nc = np.append(nc.values, new.isna().sum()).astype(float)
    rp, np_ = rc / rc.sum() + eps, nc / nc.sum() + eps
    return float(np.sum((np_ - rp) * np.log(np_ / rp)))


def feature_drift(model, ref_raw, new_raw):
    Xr, Xn = model.X(ref_raw), model.X(new_raw)
    rows = {c: psi(Xr[c], Xn[c]) for c in Xr.columns}
    rows["(puntaje del modelo)"] = psi(model.score(ref_raw), model.score(new_raw))
    out = pd.Series(rows, name="PSI").sort_values(ascending=False).to_frame()
    out["estado"] = np.select([out.PSI >= PSI_HIGH, out.PSI >= PSI_MODERATE], ["cambio fuerte", "vigilar"], "estable")
    return out


def performance(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    alert = score >= threshold
    both = 0 < y.sum() < len(y)
    return {"n": len(y), "% positivos": y.mean(),
            "ROC-AUC": roc_auc_score(y, score) if both else np.nan,
            "PR-AUC": average_precision_score(y, score) if both else np.nan,
            "recall": recall_score(y, alert) if y.sum() else np.nan,
            "precisión": precision_score(y, alert, zero_division=0) if alert.sum() else np.nan,
            "% alertado": alert.mean(), "riesgo medio": score.mean()}


def performance_by_company(model, raw):
    s = model.score(raw)
    rows = {"TOTAL": performance(raw[TARGET], s, model.thresholds["alerta"])}
    for e, g in raw.assign(_s=s).groupby("empresa"):
        rows[e] = performance(g[TARGET], g._s, model.thresholds["alerta"])
    return pd.DataFrame(rows).T


def retrain_decision(model, reference_recall, month_perf, drift, months_since_training):
    """Reglas de la propuesta: recall −5 puntos, empresas nuevas, cambio fuerte en los datos, o 3 meses."""
    reasons = []
    total = month_perf.loc["TOTAL"]
    if reference_recall - total["recall"] > RECALL_DROP:
        reasons.append(f"el recall bajó {100 * (reference_recall - total['recall']):.0f} puntos "
                       f"({reference_recall:.2f} → {total['recall']:.2f})")
    new_companies = sorted(set(month_perf.index) - {"TOTAL"} - set(model.meta.get("empresas", [])))
    if new_companies:
        reasons.append(f"empresas nuevas sin datos de entrenamiento: {', '.join(new_companies)}")
    strong = drift[drift.PSI >= PSI_HIGH].index.tolist()
    if strong:
        reasons.append(f"cambio fuerte (PSI ≥ {PSI_HIGH}) en: {', '.join(strong)}")
    calib_gap = abs(total["riesgo medio"] - total["% positivos"])
    if calib_gap > 0.05:
        reasons.append(f"descalibración: riesgo medio {total['riesgo medio']:.2f} vs real {total['% positivos']:.2f}")
    if months_since_training >= 3:
        reasons.append(f"{months_since_training} meses desde el último entrenamiento")
    return {"reentrenar": bool(reasons), "motivos": reasons}
