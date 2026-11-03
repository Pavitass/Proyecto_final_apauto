"""Las variables que la interfaz calcula desde el chat coinciden con `sql/extract.sql`.

Los valores esperados de cada caso se obtuvieron corriendo la consulta de `sql/extract.sql` sobre
estos mismos mensajes en un Postgres 15 local (tablas `conversation_sessions` y `messages` de prueba).

    .venv/bin/python tests/test_chat_features.py
"""
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from chat_features import features_from_chat  # noqa: E402


def m(de, hhmmss, texto="", adjunto=None):
    return {"de": de, "texto": texto, "adjunto": adjunto,
            "ts": datetime.strptime(f"2026-11-03 {hhmmss}", "%Y-%m-%d %H:%M:%S")}


NAN = float("nan")
CASES = {
    "consulta sencilla": (
        dict(channel="whatsapp", prior_conversations=0, prior_needed_human=0,
             messages=[m("cliente", "10:00:00", "Hola, ¿tienen envíos a Cali?"),
                       m("ia", "10:00:12", "Hola, sí hacemos envíos a todo el país."),
                       m("cliente", "10:01:00", "ok gracias"),
                       m("ia", "10:01:05", "Con gusto.")]),
        dict(hour_local=10, dow=2, w_customer_msgs=2, w_ai_msgs=2, cust_chars_mean=19.0, cust_chars_max=28,
             cust_question_marks=1, cust_repeats=0, media_image=0, media_audio=0, media_document=0, media_other=0,
             first_ai_reply_s=12.0, cust_gap_mean_s=60.0, ai_chars_mean=24.5, ai_no_info=0, cust_asks_human=0,
             cust_negative_words=0),
    ),
    "la IA escribe primero, pide asesor y el 5.º mensaje queda fuera": (
        dict(channel="instagram", prior_conversations=3, prior_needed_human=2,
             messages=[m("ia", "21:15:00", "Bienvenido, ¿en qué te ayudo?"),
                       m("cliente", "21:15:30", "Necesito hablar con un ASESOR"),
                       m("cliente", "21:16:10", "necesito hablar con un   asesor"),
                       m("cliente", "21:20:00", "???"),
                       m("ia", "21:20:05", "No tengo acceso a esa información.")]),
        dict(hour_local=21, dow=2, w_customer_msgs=3, w_ai_msgs=1, cust_chars_mean=21.0, cust_chars_max=31,
             cust_question_marks=3, cust_repeats=1, media_image=0, media_audio=0, media_document=0, media_other=0,
             first_ai_reply_s=-30.0, cust_gap_mean_s=135.0, ai_chars_mean=29.0, ai_no_info=0, cust_asks_human=1,
             cust_negative_words=0),
    ),
    "adjuntos sin texto, queja y empate de redondeo (12,25 -> 12,3)": (
        dict(channel="whatsapp", prior_conversations=1, prior_needed_human=1,
             messages=[m("cliente", "08:00:00", "", "image"),
                       m("cliente", "08:00:01", "", "audio"),
                       m("cliente", "08:00:04", "Pésimo servicio, ya van 3 días y nadie me avisa.", "application/pdf"),
                       m("cliente", "08:00:06", "x", "video"),
                       m("ia", "08:00:20", "Lamento lo ocurrido, no encontré tu pedido.")]),
        dict(hour_local=8, dow=2, w_customer_msgs=4, w_ai_msgs=0, cust_chars_mean=12.3, cust_chars_max=48,
             cust_question_marks=0, cust_repeats=0, media_image=1, media_audio=1, media_document=1, media_other=1,
             first_ai_reply_s=NAN, cust_gap_mean_s=2.0, ai_chars_mean=NAN, ai_no_info=0, cust_asks_human=0,
             cust_negative_words=1),
    ),
    "la IA no tiene la información y el cliente repite": (
        dict(channel="whatsapp", prior_conversations=0, prior_needed_human=0,
             messages=[m("cliente", "15:30:00", "Precio del 7?"),
                       m("ia", "15:30:03", "No cuento con ese dato ahora."),
                       m("cliente", "15:30:10", "Y el del 8??"),
                       m("cliente", "15:30:15", "Precio del 7?")]),
        dict(hour_local=15, dow=2, w_customer_msgs=3, w_ai_msgs=1, cust_chars_mean=12.7, cust_chars_max=13,
             cust_question_marks=4, cust_repeats=1, media_image=0, media_audio=0, media_document=0, media_other=0,
             first_ai_reply_s=3.0, cust_gap_mean_s=7.5, ai_chars_mean=29.0, ai_no_info=1, cust_asks_human=0,
             cust_negative_words=0),
    ),
}


def same(a, b):
    if isinstance(b, float) and math.isnan(b):
        return isinstance(a, float) and math.isnan(a)
    return a == b


def test_matches_extractor():
    for name, (inp, expected) in CASES.items():
        row = features_from_chat(inp["messages"], inp["channel"], inp["prior_conversations"], inp["prior_needed_human"])
        bad = {k: (row[k], v) for k, v in expected.items() if not same(row[k], v)}
        assert not bad, f"{name}: {bad}"
        assert row["channel"] == inp["channel"] and row["prior_needed_human"] == inp["prior_needed_human"]


def test_needs_a_customer_message():
    try:
        features_from_chat([m("ia", "10:00:00", "Hola")], "whatsapp")
    except ValueError:
        return
    raise AssertionError("debió rechazar una conversación sin mensajes del cliente")


if __name__ == "__main__":
    test_matches_extractor()
    test_needs_a_customer_message()
    print(f"OK: {len(CASES)} casos coinciden con sql/extract.sql")
