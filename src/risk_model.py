"""Modelo de riesgo: construcción, calibración, ajuste por tasa base y explicación.

Lo usan el notebook de modelado, la interfaz y el monitoreo, para que los tres
calculen exactamente lo mismo.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from features import build_features

DROP = ("log_ai_chars_mean",)  # company fingerprint: AUC 0.51 within companies (see notebook 01)
DAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def make_rf(**params):
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("rf", rf.set_params(**params))]).set_output(transform="pandas")


def group_splits(X, y, groups, n=5, seed=0):
    return list(StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(X, y, groups))


def calibrated(estimator, X, y, groups):
    """Isotonic calibration with folds grouped by customer, so no customer calibrates its own score."""
    return CalibratedClassifierCV(clone(estimator), method="isotonic", cv=group_splits(X, y, groups)).fit(X, y)


def shift_prior(p, prior_train, prior_new):
    """Bayes correction for label shift: P(y|x) under a new base rate, same P(x|y)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    a, b = prior_new / prior_train, (1 - prior_new) / (1 - prior_train)
    return a * p / (a * p + b * (1 - p))


def estimate_prior_em(p, prior_train, tol=1e-6, max_iter=200):
    """Saerens et al. (2002): estimate the new base rate from unlabeled predictions."""
    pi = prior_train
    for _ in range(max_iter):
        new = shift_prior(p, prior_train, pi).mean()
        if abs(new - pi) < tol:
            break
        pi = new
    return pi


def threshold_for_recall(y, p, target):
    """Highest threshold whose recall is still >= target (highest precision at that recall)."""
    order = np.argsort(-p)
    ys, ps = np.asarray(y)[order], p[order]
    recall = np.cumsum(ys) / ys.sum()
    return float(ps[np.argmax(recall >= target)])


class RiskModel:
    """Artefacto desplegable: modelo calibrado + modelo para explicar + umbrales + tasa base."""

    def __init__(self, model, explain_model, thresholds, prior_train, features, meta=None):
        self.model, self.explain_model = model, explain_model
        self.thresholds, self.prior_train, self.features = thresholds, prior_train, features
        self.meta = meta or {}

    def X(self, raw):
        return build_features(raw, DROP)[self.features]

    def score(self, raw):
        """Probability calibrated to the training base rate; alert thresholds apply to this score."""
        return self.model.predict_proba(self.X(raw))[:, 1]

    def risk(self, raw, base_rate=None):
        # base_rate only for a genuine label shift inside the same companies; a change in company
        # mix is covariate shift and the model already follows it (notebook 02, section 4).
        s = self.score(raw)
        shown = s if base_rate is None else shift_prior(s, self.prior_train, base_rate)
        level = np.where(s >= self.thresholds["alerta"], "rojo", np.where(s >= self.thresholds["vigilar"], "amarillo", "verde"))
        return pd.DataFrame({"score": s, "riesgo": shown, "nivel": level}, index=raw.index)

    def shap_values(self, raw):
        import shap
        X = self.explain_model[:-1].transform(self.X(raw))
        sv = shap.TreeExplainer(self.explain_model[-1])(X)
        return pd.DataFrame(sv.values[:, :, 1], columns=X.columns, index=raw.index), float(np.ravel(sv.base_values)[1])


def train_risk_model(raw, params, meta=None, recall_targets=(0.80, 0.95), seed=2):
    """Entrena el artefacto completo. Los umbrales salen de predicciones fuera de pliegue
    (validación anidada agrupada por cliente), nunca de las mismas filas con que se ajustó el modelo."""
    from features import TARGET
    X, y, g = build_features(raw, DROP), raw[TARGET].values, raw.cliente.values
    oof = np.zeros(len(raw))
    for a, b in group_splits(X, y, g, seed=seed):
        oof[b] = calibrated(make_rf(**params), X.iloc[a], y[a], g[a]).predict_proba(X.iloc[b])[:, 1]
    thresholds = {"alerta": threshold_for_recall(y, oof, recall_targets[0]),
                  "vigilar": threshold_for_recall(y, oof, recall_targets[1])}
    meta = {"params": params, "n": len(raw), "fecha_min": str(raw.fecha.min().date()),
            "fecha_max": str(raw.fecha.max().date()), "empresas": sorted(raw.empresa.unique()), **(meta or {})}
    model = RiskModel(model=calibrated(make_rf(**params), X, y, g), explain_model=make_rf(**params).fit(X, y),
                      thresholds=thresholds, prior_train=float(y.mean()), features=list(X.columns), meta=meta)
    return model, oof


# --- Explanation in plain language -------------------------------------------------------------

GROUPS = {
    "historial": ["prior_human_rate", "prior_needed_human", "prior_conversations", "has_history"],
    "largo": ["log_cust_chars_mean", "log_cust_chars_max"],
    "preguntas": ["cust_question_marks"],
    "mensajes": ["w_customer_msgs", "single_customer_msg"],
    "respuesta": ["log_first_ai_reply_s", "no_ai_reply"],
    "ritmo": ["log_cust_gap_mean_s"],
    "hora": ["hour_sin", "hour_cos"],
    "dia": ["weekend"] + [f"dow_{d}" for d in range(1, 8)],
    "canal": ["instagram"],
    "adjuntos": ["media_image", "media_audio", "media_document", "media_other", "has_media"],
    "repite": ["cust_repeats"],
    "sin_info": ["ai_no_info"],
    "asesor": ["cust_asks_human"],
    "queja": ["cust_negative_words"],
    "largo_ia": ["log_ai_chars_mean"],
}


def _dur(s):
    s = abs(s)
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def _plural(n, one, many):
    return f"{n} {one if n == 1 else many}"


def describe(group, r):
    """One sentence for a group of features, written from the raw row."""
    if group == "historial":
        n, k = int(r.prior_conversations), int(r.prior_needed_human)
        if n == 0:
            return "Es su primera conversación registrada"
        if n == 1:
            return f"Tuvo 1 conversación anterior y {'necesitó' if k else 'no necesitó'} a una persona"
        return f"De sus {n} conversaciones anteriores, {k} necesitaron a una persona"
    if group == "largo":
        # Median customer message is ~25 characters (notebook 01).
        size = "muy cortos" if r.cust_chars_mean < 15 else "de largo habitual" if r.cust_chars_mean <= 40 else "largos"
        return f"Sus mensajes son {size} ({r.cust_chars_mean:.0f} caracteres en promedio)"
    if group == "preguntas":
        q = int(r.cust_question_marks)
        return f"Hizo {_plural(q, 'pregunta', 'preguntas')} explícitas (con «?»)" if q else "No hizo preguntas explícitas"
    if group == "mensajes":
        return f"Escribió {_plural(int(r.w_customer_msgs), 'mensaje', 'mensajes')} en los primeros 4"
    if group == "respuesta":
        if pd.isna(r.first_ai_reply_s):
            return "La atención automática no alcanzó a responder en los primeros 4 mensajes"
        if r.first_ai_reply_s < 0:
            return "La atención automática escribió primero"
        return f"La primera respuesta automática tardó {_dur(r.first_ai_reply_s)}"
    if group == "ritmo":
        return "Envió un solo mensaje" if pd.isna(r.cust_gap_mean_s) else f"Entre sus mensajes pasaron {_dur(r.cust_gap_mean_s)} en promedio"
    if group == "hora":
        return f"Escribió a las {int(r.hour_local)}:00"
    if group == "dia":
        return f"Escribió un {DAYS[int(r.dow) - 1]}"
    if group == "canal":
        return f"Llegó por {'Instagram' if r.channel == 'instagram' else 'WhatsApp'}"
    if group == "adjuntos":
        parts = [_plural(int(r[c]), one, many) for c, one, many in
                 [("media_image", "foto", "fotos"), ("media_audio", "audio", "audios"),
                  ("media_document", "documento", "documentos"), ("media_other", "otro adjunto", "otros adjuntos")] if r[c] > 0]
        return "Envió " + ", ".join(parts) if parts else "No envió adjuntos"
    if group == "repite":
        return f"Repitió el mismo mensaje {_plural(int(r.cust_repeats), 'vez', 'veces')}" if r.cust_repeats else "No repitió mensajes"
    if group == "sin_info":
        return "La IA dijo que no tenía la información" if r.ai_no_info else "La IA no dijo que le faltara información"
    if group == "asesor":
        return "Pidió hablar con un asesor" if r.cust_asks_human else "No pidió un asesor"
    if group == "queja":
        return "Usó palabras de queja" if r.cust_negative_words else "No usó palabras de queja"
    if group == "largo_ia":
        return f"Las respuestas automáticas tuvieron {r.ai_chars_mean:.0f} caracteres en promedio"
    return group


def explain(raw_row, shap_row, top=4, min_effect=0.01):
    """Top reasons, as (sentence, effect on probability, 'sube'|'baja')."""
    effects = {g: shap_row[[c for c in cols if c in shap_row.index]].sum() for g, cols in GROUPS.items()
               if any(c in shap_row.index for c in cols)}
    ranked = sorted(effects.items(), key=lambda kv: -abs(kv[1]))
    return [(describe(g, raw_row), float(v), "sube" if v > 0 else "baja") for g, v in ranked[:top] if abs(v) >= min_effect]
