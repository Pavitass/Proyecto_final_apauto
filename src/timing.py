"""Tiempos de desenlace medidos en producción (`sql/timing.sql`) unidos al dataset anonimizado.

La consulta no devuelve identificadores, así que cada fila se empareja con el CSV por sus propias
variables. Si dos conversaciones comparten todas las variables, se toma el caso más desfavorable
(cualquier escalación dentro de la ventana marca a todas).
"""
import glob

import numpy as np
import pandas as pd

KEY = ["fecha", "channel", "hour_local", "dow", "w_customer_msgs", "w_ai_msgs", "cust_chars_mean",
       "cust_chars_max", "cust_question_marks", "first_ai_reply_s", "cust_gap_mean_s", "ai_chars_mean"]
REL = ["esc_rel_s", "human_rel_s", "last_msg_rel_s"]


def _key(df):
    def fmt(v):
        if pd.isna(v):
            return ""
        if isinstance(v, pd.Timestamp):
            return str(v.date())
        return f"{v:.1f}" if isinstance(v, float) else str(v)
    return df[KEY].map(fmt).agg("|".join, axis=1)


def load_timing(pattern):
    t = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pattern))], ignore_index=True)
    t["_k"] = _key(t)
    return t.groupby("_k").agg(w_msgs_after_esc=("w_msgs_after_esc", "max"), **{c: (c, "min") for c in REL})


def attach(raw, timing):
    """Agrega a cada fila del dataset si hubo escalación dentro de la ventana y los segundos entre el
    4.º mensaje (momento de la predicción) y cada desenlace. `matched` = False si no se encontró."""
    out = raw.assign(_k=_key(raw)).join(timing, on="_k").drop(columns="_k")
    out["matched"] = out.w_msgs_after_esc.notna()
    out["esc_in_window"] = out.w_msgs_after_esc.fillna(0) > 0
    out["need_rel_s"] = out[["esc_rel_s", "human_rel_s"]].min(axis=1)
    return out


def margin_summary(rel_s):
    """Margen en minutos entre la predicción y el desenlace."""
    m = pd.Series(rel_s).dropna() / 60
    return {"n": len(m), "mediana_min": m.median(), "p25_min": m.quantile(.25), "p75_min": m.quantile(.75),
            "p90_min": m.quantile(.9), "% antes": (m > 0).mean(), "% ≥1 min": (m >= 1).mean(),
            "% ≥5 min": (m >= 5).mean()}
