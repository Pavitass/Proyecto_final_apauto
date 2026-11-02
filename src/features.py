"""Preprocesamiento compartido por el EDA, el modelado y la interfaz.

Todo lo que se aprende de los datos (imputación, escalado) va en un Pipeline de
sklearn que se ajusta solo con entrenamiento; aquí solo hay transformaciones fila
a fila, que no pueden filtrar información del conjunto de prueba.
"""
import numpy as np
import pandas as pd

TARGET = "needed_human"
DOMINANT = "E09"
ID_COLS = ["empresa", "cliente", "fecha"]

LOG_COLS = ["cust_chars_mean", "cust_chars_max", "ai_chars_mean", "first_ai_reply_s", "cust_gap_mean_s"]
COUNT_COLS = ["cust_question_marks", "cust_repeats", "media_image", "media_audio",
              "media_document", "media_other", "prior_conversations", "prior_needed_human"]
FLAG_COLS = ["ai_no_info", "cust_asks_human", "cust_negative_words"]


def load(path):
    df = pd.read_csv(path, parse_dates=["fecha"])
    df["mes"] = df.fecha.dt.to_period("M").astype(str)
    return df


def signed_log(s):
    # first_ai_reply_s can be negative (the bot wrote first), so a plain log1p is not enough.
    return np.sign(s) * np.log1p(s.abs())


def build_features(df, drop=()):
    """Devuelve la matriz X (sin imputar ni escalar) a partir del CSV crudo."""
    X = pd.DataFrame(index=df.index)
    X["w_customer_msgs"] = df.w_customer_msgs
    # w_ai_msgs is omitted: within the fixed 4-message window it is 4 - w_customer_msgs.
    for c in LOG_COLS:
        X[f"log_{c}"] = signed_log(df[c])
    # Missingness is informative: no AI reply in the window, or a single customer message.
    X["no_ai_reply"] = df.first_ai_reply_s.isna().astype(int)
    X["single_customer_msg"] = df.cust_gap_mean_s.isna().astype(int)
    for c in COUNT_COLS:
        X[c] = np.log1p(df[c])
    X["prior_human_rate"] = (df.prior_needed_human / df.prior_conversations).where(df.prior_conversations > 0)
    X["has_history"] = (df.prior_conversations > 0).astype(int)
    for c in FLAG_COLS:
        X[c] = df[c]
    X["has_media"] = (df[["media_image", "media_audio", "media_document", "media_other"]].sum(axis=1) > 0).astype(int)
    X["hour_sin"] = np.sin(2 * np.pi * df.hour_local / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df.hour_local / 24)
    X["weekend"] = df.dow.isin([6, 7]).astype(int)
    for d in range(1, 8):
        X[f"dow_{d}"] = (df.dow == d).astype(int)
    X["instagram"] = (df.channel == "instagram").astype(int)
    return X.drop(columns=list(drop))


def time_split(df, cutoff="2026-10-01"):
    return df[df.fecha < cutoff], df[df.fecha >= cutoff]
