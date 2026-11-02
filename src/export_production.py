"""Exporta el modelo de producción (`riesgo_v1_prod`) a JSON para evaluarlo en el backend.

`riesgo_v1.joblib` (5 bosques calibrados x 300 árboles + un bosque para explicar, 13 MB) es el
artefacto de investigación. En producción basta un solo bosque más chico más un mapa isotónico,
con la misma familia de variables y parámetros:

- `RandomForestClassifier` con los parámetros de `models/best_params.json` y menos árboles;
- calibración isotónica ajustada con predicciones fuera de pliegue agrupadas por cliente (ningún
  cliente calibra su propio puntaje), **centrada**: cada escalón de la isotónica se reduce a su punto
  medio y se interpola linealmente entre ellos. Un solo mapa isotónico deja ~25 escalones y empata a
  cientos de conversaciones, lo que bajaba el PR-AUC de octubre 0,02; la versión centrada es
  estrictamente creciente, conserva el orden del bosque y calibra igual (Brier 0,184);
- umbrales (alerta = recall 0,80, vigilar = recall 0,95) con predicciones fuera de pliegue anidadas,
  igual que `train_risk_model`;
- razones por el método de caminos de Saabas sobre el mismo bosque (no hace falta un bosque aparte).

Uso:
    .venv/bin/python -I src/export_production.py            # exporta y escribe el reporte
    .venv/bin/python -I src/export_production.py --explore  # compara tamaños de bosque

Escribe `models/riesgo_v1_prod.json`, `models/riesgo_v1_prod_fixtures.json` (filas crudas ->
variables, puntaje, nivel y razones esperadas, para la prueba de paridad del backend) y
`models/riesgo_v1_prod_metrics.json`.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import DOMINANT, TARGET, build_features, load  # noqa: E402
from risk_model import DROP, GROUPS, calibrated, group_splits, make_rf, threshold_for_recall  # noqa: E402

VERSION = "riesgo_v1_prod"
N_TREES = 60
MAX_JSON_BYTES = 2_000_000
RAW_FIELDS = ["channel", "hour_local", "dow", "w_customer_msgs", "w_ai_msgs", "cust_chars_mean", "cust_chars_max",
              "cust_question_marks", "cust_repeats", "media_image", "media_audio", "media_document", "media_other",
              "first_ai_reply_s", "cust_gap_mean_s", "ai_chars_mean", "ai_no_info", "cust_asks_human",
              "cust_negative_words", "prior_conversations", "prior_needed_human"]


class ProdModel:
    """Un bosque con imputación por mediana + mapa isotónico, evaluado igual que en el backend."""

    def __init__(self, imputer, rf, iso_x, iso_y, features):
        self.imputer, self.rf, self.iso_x, self.iso_y, self.features = imputer, rf, iso_x, iso_y, features

    def raw_proba(self, X):
        return self.rf.predict_proba(self.imputer.transform(X[self.features]))[:, 1]

    def score(self, X):
        return np.interp(self.raw_proba(X), self.iso_x, self.iso_y)


def fit_forest(X, y, params, n_trees, seed=0):
    imp = SimpleImputer(strategy="median").fit(X)
    rf = RandomForestClassifier(n_estimators=n_trees, class_weight="balanced", n_jobs=-1, random_state=seed, **params)
    return imp, rf.fit(imp.transform(X), y)


def centered_isotonic(iso):
    """Each plateau of the fitted isotonic step function collapses to its midpoint; np.interp between
    the midpoints (clipped at the ends) is strictly increasing inside the range."""
    x, yv = iso.X_thresholds_, iso.y_thresholds_
    xs, ys, i = [], [], 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and yv[j + 1] == yv[i]:
            j += 1
        xs.append(float((x[i] + x[j]) / 2))
        ys.append(float(yv[i]))
        i = j + 1
    return np.array(xs), np.array(ys)


def fit_prod(X, y, g, params, n_trees):
    """Bosque con todas las filas; el mapa isotónico sale de predicciones fuera de pliegue."""
    oof = np.zeros(len(y))
    for a, b in group_splits(X, y, g):
        imp, rf = fit_forest(X.iloc[a], y[a], params, n_trees)
        oof[b] = rf.predict_proba(imp.transform(X.iloc[b]))[:, 1]
    iso_x, iso_y = centered_isotonic(IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof, y))
    imp, rf = fit_forest(X, y, params, n_trees)
    return ProdModel(imp, rf, iso_x, iso_y, list(X.columns))


def oof_scores(X, y, g, params, n_trees, seed=2):
    """Puntajes calibrados fuera de pliegue (validación anidada agrupada por cliente)."""
    oof = np.zeros(len(y))
    for a, b in group_splits(X, y, g, seed=seed):
        m = fit_prod(X.iloc[a], y[a], g[a], params, n_trees)
        oof[b] = m.score(X.iloc[b])
    return oof


def metrics(y, s):
    return {"PR-AUC": float(average_precision_score(y, s)), "ROC-AUC": float(roc_auc_score(y, s))}


def export_trees(rf):
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        p1 = v[:, 1] / v.sum(axis=1)
        leaf = t.children_left == -1
        trees.append({
            "feature": [int(f) if not lf else -1 for f, lf in zip(t.feature, leaf)],
            # Leaves carry no threshold; 0 keeps the arrays aligned and short.
            "threshold": [float(th) if not lf else 0 for th, lf in zip(t.threshold, leaf)],
            "left": [int(c) for c in t.children_left],
            "right": [int(c) for c in t.children_right],
            "value": [round(float(p), 12) for p in p1],
        })
    return trees


# --- Pure-Python evaluation of the exported JSON (same algorithm as the TypeScript port) -------

def f32(x):
    return float(np.float32(x))


def eval_json(model, feats):
    """feats: dict feature -> value (None = missing). Returns raw proba, score, contributions."""
    x = [model["imputer_medians"][f] if feats[f] is None else feats[f] for f in model["features"]]
    x32 = [f32(v) for v in x]
    total, contrib = 0.0, [0.0] * len(x)
    for t in model["trees"]:
        node = 0
        while t["left"][node] != -1:
            f = t["feature"][node]
            nxt = t["left"][node] if x32[f] <= t["threshold"][node] else t["right"][node]
            contrib[f] += t["value"][nxt] - t["value"][node]
            node = nxt
        total += t["value"][node]
    raw = total / len(model["trees"])
    iso = model["isotonic"]
    score = float(np.interp(raw, iso["x"], iso["y"]))
    n = len(model["trees"])
    return raw, score, [c / n for c in contrib]


def level(score, th):
    return "high" if score >= th["alert"] else "watch" if score >= th["watch"] else "low"


def group_weights(model, contrib):
    idx = {f: i for i, f in enumerate(model["features"])}
    return {grp: sum(contrib[idx[c]] for c in cols if c in idx) for grp, cols in model["reason_groups"].items()}


def nan_to_none(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def pick_fixtures(test, n=60, seed=7):
    """Muestra estratificada de octubre + casos borde (sin respuesta, bot primero, Instagram, historial)."""
    rng = np.random.RandomState(seed)
    edge = pd.concat([
        test[test.first_ai_reply_s.isna()].head(4),
        test[test.first_ai_reply_s < 0].head(4),
        test[test.cust_gap_mean_s.isna()].head(3),
        test[test.channel == "instagram"].head(4),
        test[test.prior_conversations > 5].head(3),
        test[(test.media_image + test.media_audio + test.media_document + test.media_other) > 0].head(4),
        test[(test.ai_no_info + test.cust_asks_human + test.cust_negative_words + test.cust_repeats) > 0].head(4),
        test[test.dow.isin([6, 7])].head(2),
    ])
    rest = test.drop(edge.index, errors="ignore")
    fill = rest.iloc[rng.choice(len(rest), n - len(edge.index.unique()), replace=False)]
    return pd.concat([edge[~edge.index.duplicated()], fill])


def brier(y, s):
    return float(np.mean((np.asarray(s) - y) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--trees", type=int, default=N_TREES)
    a = ap.parse_args()

    params = json.load(open(ROOT / "models" / "best_params.json"))
    raw = load(ROOT / "data" / "orvyz_conversaciones_riesgo.csv")
    raw = raw[raw.empresa != DOMINANT].reset_index(drop=True)
    X, y, g = build_features(raw, DROP), raw[TARGET].values, raw.cliente.values
    tr, te = (raw.fecha < "2026-10-01").values, (raw.mes == "2026-10").values

    # Reference: riesgo_v1 (5 calibrated forests x 300 trees) on the same temporal split.
    ref = calibrated(make_rf(**params), X[tr], y[tr], g[tr]).predict_proba(X[te])[:, 1]
    ref_m = {**metrics(y[te], ref), "Brier": brier(y[te], ref), "riesgo medio": float(ref.mean())}
    print("riesgo_v1 temporal:", ref_m, flush=True)

    sizes = [20, 40, 60, 80, 100] if a.explore else [a.trees]
    for n in sizes:
        m = fit_prod(X[tr], y[tr], g[tr], params, n)
        st = m.score(X[te])
        tm = {**metrics(y[te], st), "Brier": brier(y[te], st), "riesgo medio": float(st.mean()), "real": float(y[te].mean())}
        size = len(json.dumps(export_trees(m.rf), separators=(",", ":")))
        print(f"{n} trees temporal: {tm} | trees JSON ~{size / 1e6:.2f} MB (fit with data <= Sep)", flush=True)
    if a.explore:
        return

    temporal = tm
    oof = oof_scores(X, y, g, params, a.trees)
    thresholds = {"alert": threshold_for_recall(y, oof, 0.80), "watch": threshold_for_recall(y, oof, 0.95)}
    final = fit_prod(X, y, g, params, a.trees)
    model = {
        "version": VERSION,
        "features": final.features,
        "imputer_medians": {f: float(v) for f, v in zip(final.features, final.imputer.statistics_)},
        "trees": export_trees(final.rf),
        "isotonic": {"x": [float(v) for v in final.iso_x], "y": [float(v) for v in final.iso_y]},
        "thresholds": thresholds,
        "prior_train": float(y.mean()),
        "reason_groups": {k: [c for c in v if c in final.features] for k, v in GROUPS.items()
                          if any(c in final.features for c in v)},
        "meta": {"n": int(len(raw)), "trees": a.trees, "params": params, "fecha_min": str(raw.fecha.min().date()),
                 "fecha_max": str(raw.fecha.max().date()),
                 "calibration": "centered isotonic (plateau midpoints, linear interpolation, clipped) on out-of-fold forest probabilities grouped by customer"},
    }
    text = json.dumps(model, separators=(",", ":"), allow_nan=False)
    assert len(text) <= MAX_JSON_BYTES, f"JSON too large: {len(text)} bytes"
    (ROOT / "models" / f"{VERSION}.json").write_text(text)

    # The JSON evaluator must match sklearn before the backend relies on it.
    full = final.score(X)
    fx = pick_fixtures(raw[te])
    rows, worst = [], 0.0
    for i, r in fx.iterrows():
        feats = {f: nan_to_none(X.at[i, f]) for f in final.features}
        _, s, contrib = eval_json(model, feats)
        worst = max(worst, abs(s - full[i]))
        gw = group_weights(model, contrib)
        rows.append({"raw": {k: nan_to_none(r[k]) for k in RAW_FIELDS}, "features": feats, "score": float(full[i]),
                     "level": level(float(full[i]), thresholds), "group_weights": gw})
    assert worst <= 1e-9, f"JSON evaluator differs from sklearn by {worst}"
    (ROOT / "models" / f"{VERSION}_fixtures.json").write_text(
        json.dumps({"version": VERSION, "rows": rows}, indent=1, ensure_ascii=False, allow_nan=False))

    oof_m = metrics(y, oof)
    alert = oof >= thresholds["alert"]
    report = {
        "version": VERSION, "trees": a.trees, "json_bytes": len(text),
        "temporal_octubre": {"riesgo_v1": ref_m, VERSION: temporal,
                             "diff": {k: temporal[k] - ref_m[k] for k in ("PR-AUC", "ROC-AUC")}},
        "fuera_de_pliegue_final": {**oof_m, "recall_alerta": float(y[alert].sum() / y.sum()),
                                   "precision_alerta": float(y[alert].mean()), "pct_alertado": float(alert.mean())},
        "thresholds": thresholds, "prior_train": model["prior_train"],
        "max_abs_diff_json_vs_sklearn": worst,
    }
    (ROOT / "models" / f"{VERSION}_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
