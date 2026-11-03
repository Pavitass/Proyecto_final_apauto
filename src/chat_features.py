"""Variables del modelo calculadas desde los mensajes de una conversación.

Replica, mensaje a mensaje, la definición de `sql/extract.sql`, para que la interfaz (y una
integración en la plataforma) produzcan exactamente la misma fila que el extractor:

- ventana = los primeros 4 mensajes, en orden de hora, antes de cualquier mensaje humano;
- textos normalizados como en el SQL: `lower(regexp_replace(texto, '\\s+', ' ', 'g'))`;
- redondeos a un decimal como `round(numeric, 1)` de Postgres (mitad hacia arriba), no como `round`
  de Python (mitad al par).

Cada mensaje es un dict con `de` ("cliente" o "ia"), `texto`, `ts` (datetime en hora de Colombia) y
opcionalmente `adjunto` ("image", "audio", "document", "video" o None).
"""
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

WINDOW = 4
AI_NO_INFO = re.compile(r"(no tengo|no cuento con|no dispongo|no puedo ayudarte|no tengo acceso|no encontr)")
ASKS_HUMAN = re.compile(r"(asesor|humano|una persona|agente real|hablar con alguien)")
NEGATIVE = re.compile(r"(queja|reclamo|pesimo|pésimo|terrible|molest|estafa|nadie responde)")
RAW_FIELDS = ["channel", "hour_local", "dow", "w_customer_msgs", "w_ai_msgs", "cust_chars_mean", "cust_chars_max",
              "cust_question_marks", "cust_repeats", "media_image", "media_audio", "media_document", "media_other",
              "first_ai_reply_s", "cust_gap_mean_s", "ai_chars_mean", "ai_no_info", "cust_asks_human",
              "cust_negative_words", "prior_conversations", "prior_needed_human"]


def _norm(text):
    return re.sub(r"\s+", " ", text).lower()


def _round1(x):
    return float(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _media_kind(media):
    """Same buckets as the ILIKE filters of the extractor."""
    m = (media or "").lower()
    if not m or m == "text":
        return None
    if m.startswith("image"):
        return "media_image"
    if m.startswith(("audio", "voice")):
        return "media_audio"
    if m.startswith(("document", "application")):
        return "media_document"
    return "media_other"


def features_from_chat(messages, channel, prior_conversations=0, prior_needed_human=0):
    """Una fila cruda (mismas columnas que el CSV) desde los mensajes. La hora y el día salen del
    primer mensaje, que es cuando arranca la sesión."""
    msgs = sorted(messages, key=lambda m: m["ts"])[:WINDOW]
    cust = [m for m in msgs if m["de"] == "cliente"]
    ai = [m for m in msgs if m["de"] == "ia"]
    if not cust:
        raise ValueError("La conversación necesita al menos un mensaje del cliente.")
    texts = [m.get("texto") or "" for m in cust]
    norms = [_norm(t) for t in texts]
    repeats = sum(1 for i, n in enumerate(norms) if n and n in norms[:i])
    gaps = [(b["ts"] - a["ts"]).total_seconds() for a, b in zip(cust, cust[1:])]
    media = {k: 0 for k in ("media_image", "media_audio", "media_document", "media_other")}
    for m in cust:
        kind = _media_kind(m.get("adjunto"))
        if kind:
            media[kind] += 1
    first_reply = (ai[0]["ts"] - cust[0]["ts"]).total_seconds() if ai else np.nan
    start = msgs[0]["ts"]
    return {
        "channel": channel, "hour_local": start.hour, "dow": start.isoweekday(),
        "w_customer_msgs": len(cust), "w_ai_msgs": len(ai),
        "cust_chars_mean": _round1(np.mean([len(t) for t in texts])), "cust_chars_max": max(len(t) for t in texts),
        "cust_question_marks": sum(t.count("?") for t in texts), "cust_repeats": repeats, **media,
        "first_ai_reply_s": _round1(first_reply) if ai else np.nan,
        "cust_gap_mean_s": _round1(np.mean(gaps)) if gaps else np.nan,
        "ai_chars_mean": _round1(np.mean([len(m.get("texto") or "") for m in ai])) if ai else np.nan,
        "ai_no_info": int(any(AI_NO_INFO.search(_norm(m.get("texto") or "")) for m in ai)),
        "cust_asks_human": int(any(ASKS_HUMAN.search(n) for n in norms)),
        "cust_negative_words": int(any(NEGATIVE.search(n) for n in norms)),
        "prior_conversations": int(prior_conversations), "prior_needed_human": int(prior_needed_human),
    }
