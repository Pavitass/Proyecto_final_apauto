"""Interfaz: riesgo de que una conversación necesite a una persona, con semáforo y razones.

    .venv/bin/streamlit run app/app.py
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from chat_features import RAW_FIELDS, features_from_chat  # noqa: E402
from features import DOMINANT, TARGET, load  # noqa: E402
from risk_model import DAYS, explain  # noqa: E402

LEVELS = {
    "rojo": ("#e34948", "Alerta", "Pasar a una persona del equipo ahora."),
    "amarillo": ("#eda100", "Vigilar", "Dejar con la atención automática, pero ponerla arriba en la cola."),
    "verde": ("#1baf7a", "Bajo", "La atención automática probablemente la resuelve sola."),
}
SENDERS = {"cliente": "Cliente", "ia": "Atención automática"}
ATTACHMENTS = {"": "Sin adjunto", "image": "Foto", "audio": "Audio", "document": "Documento", "video": "Video u otro"}
REASONS_LIMIT = 500  # SHAP takes ~11 ms per row; beyond this the ranking still works, without reasons

# Anticipation is measured in production (sql/timing.sql, notebook Parte 2 §3.c); see documento técnico §3.7.
ANTICIPATION = {"antes": "100 %", "mediana": "27 min", "5min": "8 de cada 10"}

# Conversaciones inventadas para probar la interfaz; no son datos reales.
PRESETS = {
    "verde": dict(
        label="Cliente frecuente, pedido de rutina", channel="whatsapp", dow=4, prior=6, prior_h=0,
        bubbles=[("cliente", "16:20:00", "Buenas tardes", ""),
                 ("ia", "16:20:05", "¡Hola de nuevo! ¿Te ayudo con un pedido?", ""),
                 ("cliente", "16:20:20", "El pedido de siempre", ""),
                 ("ia", "16:20:25", "Listo, te comparto el enlace para confirmar tu pedido.", "")]),
    "rojo": dict(
        label="Pedido que no llega", channel="instagram", dow=5, prior=3, prior_h=3,
        bubbles=[("cliente", "19:40:00", "Buenas noches, hice un pedido la semana pasada y todavía no me llega. "
                                          "Ya pagué y nadie me da razón, necesito saber qué pasó con mi pedido", ""),
                 ("cliente", "19:52:00", "", "image"),
                 ("cliente", "20:05:00", "Hola??", ""),
                 ("ia", "20:05:03", "Hola, con gusto te ayudo. ¿Me compartes el número de tu pedido?", "")]),
    "dudoso": dict(
        label="Cambio de talla (dudoso)", channel="whatsapp", dow=3, prior=1, prior_h=0,
        bubbles=[("cliente", "15:20:00", "Hola, quiero cambiar la talla de una prenda que compré", ""),
                 ("ia", "15:20:05", "Hola, claro. ¿Me indicas el número de pedido y la talla que necesitas?", ""),
                 ("cliente", "15:24:00", "Es el pedido 4512, necesito una talla M en vez de S", ""),
                 ("ia", "15:24:04", "Gracias. Ya revisé el pedido; te cuento cómo hacer el cambio.", "")]),
}

st.set_page_config(page_title="Riesgo de conversación", layout="wide")


@st.cache_resource
def get_model():
    return joblib.load(ROOT / "models" / "riesgo_v1.joblib")


@st.cache_data
def get_metrics():
    return json.load(open(ROOT / "models" / "metrics.json", encoding="utf-8"))


@st.cache_data
def get_examples(per_level=8):
    df = load(ROOT / "data" / "orvyz_conversaciones_riesgo.csv")
    oct_ = df[(df.empresa != DOMINANT) & (df.mes == "2026-10")]
    lv = get_model().risk(oct_)
    picks = [oct_.loc[lv.index[lv.nivel == n]].sample(per_level, random_state=1) for n in LEVELS]
    ex = pd.concat(picks).sample(frac=1, random_state=2).reset_index(drop=True)
    ex["etiqueta"] = [f"Ejemplo {i + 1:02d} · {r.empresa} · {channel_name(r.channel)} · "
                      f"{DAYS[r.dow - 1]} {r.hour_local}:00" for i, r in ex.iterrows()]
    return ex


def channel_name(c):
    return "Instagram" if c == "instagram" else "WhatsApp"


def level_of(score, th):
    return "rojo" if score >= th["alerta"] else "amarillo" if score >= th["vigilar"] else "verde"


# --- Result view ---------------------------------------------------------------------------------

def badge(level):
    color, name, action = LEVELS[level]
    return (f"<div style='display:flex;align-items:center;gap:12px;padding:14px 16px;border-radius:12px;"
            f"border:1px solid {color};background:{color}1a'>"
            f"<div style='width:22px;height:22px;border-radius:50%;background:{color};flex:none'></div>"
            f"<div><div style='font-weight:600;font-size:1.05rem'>{name}</div>"
            f"<div style='opacity:.8;font-size:.9rem'>{action}</div></div></div>")


def gauge(score, th):
    """Semicircular meter: green / yellow / red zones split at the two thresholds, needle at the score."""
    cx, cy, r = 120, 112, 92

    def pt(v, rad=r):
        a = np.pi * (1 - v)
        return cx + rad * np.cos(a), cy - rad * np.sin(a)

    def arc(a, b, color):
        (x1, y1), (x2, y2) = pt(a), pt(b)
        return (f"<path d='M{x1:.1f},{y1:.1f} A{r},{r} 0 0 1 {x2:.1f},{y2:.1f}' fill='none' stroke='{color}' "
                f"stroke-width='18' stroke-linecap='butt'/>")

    v, a = th["vigilar"], th["alerta"]
    nx, ny = pt(score, r - 26)
    ticks = "".join(f"<text x='{pt(t, r + 16)[0]:.0f}' y='{pt(t, r + 16)[1] + 4:.0f}' font-size='10' "
                    f"text-anchor='middle' fill='currentColor' opacity='.65'>{t:.0%}</text>" for t in (v, a))
    color = LEVELS[level_of(score, th)][0]
    return (f"<svg viewBox='0 0 240 136' style='width:100%;max-width:300px;display:block;margin:4px auto 8px' "
            f"role='img' aria-label='Riesgo {score:.0%}'>"
            f"{arc(0, v, LEVELS['verde'][0])}{arc(v, a, LEVELS['amarillo'][0])}{arc(a, 1, LEVELS['rojo'][0])}"
            f"{ticks}<line x1='{cx}' y1='{cy}' x2='{nx:.1f}' y2='{ny:.1f}' stroke='currentColor' stroke-width='3' "
            f"stroke-linecap='round'/><circle cx='{cx}' cy='{cy}' r='6' fill='currentColor'/>"
            f"<text x='{cx}' y='{cy + 22}' font-size='20' font-weight='700' text-anchor='middle' fill='{color}'>"
            f"{score:.0%}</text></svg>")


def reasons_html(reasons):
    if not reasons:
        return "Ninguna señal se aleja del promedio."
    rows = []
    for text, effect, direction in reasons:
        pts = round(100 * abs(effect))
        arrow, color = ("▲", "#e34948") if direction == "sube" else ("▼", "#1baf7a")
        rows.append(f"<div style='padding:6px 0;border-bottom:1px solid rgba(128,128,128,.2)'>"
                    f"<span style='color:{color};font-weight:700'>{arrow}</span> {text} "
                    f"<span style='opacity:.65'>· {direction} {pts} punto{'' if pts == 1 else 's'}</span></div>")
    return "".join(rows)


def show_result(row, truth=None):
    model = get_model()
    r = model.risk(row).iloc[0]
    sv, _ = model.shap_values(row)
    reasons = explain(row.iloc[0], sv.iloc[0], top=5)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("**Riesgo de necesitar una persona**")
        st.markdown(gauge(r.score, model.thresholds), unsafe_allow_html=True)
        st.markdown(badge(r.nivel), unsafe_allow_html=True)
        if truth is not None:
            with st.expander("¿Qué pasó en realidad?"):
                st.write("**Necesitó a una persona.**" if truth else "**La atención automática la resolvió sola.**")
    with right:
        st.markdown("**Por qué**")
        st.markdown(reasons_html(reasons), unsafe_allow_html=True)
        st.caption("Cada razón agrupa variables relacionadas (valores SHAP). Los puntos son cuánto mueve la "
                   "razón la probabilidad respecto a una conversación promedio.")


# --- Conversation as a chat ----------------------------------------------------------------------

def load_preset(name):
    p = PRESETS[name]
    st.session_state.update(chat_channel=p["channel"], chat_dow=p["dow"], chat_prior=p["prior"], chat_prior_h=p["prior_h"])
    for i, (de, hora, texto, adj) in enumerate(p["bubbles"]):
        st.session_state.update({f"b{i}_de": de, f"b{i}_hora": hora, f"b{i}_texto": texto, f"b{i}_adj": adj})


def parse_time(text):
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(text.strip(), fmt).time()
        except ValueError:
            pass
    return None


def chat_editor():
    """Four bubbles plus context; returns the raw row computed from them, or None with the problems shown."""
    if "b0_de" not in st.session_state:
        load_preset("verde")
    st.markdown("**Cargar un ejemplo** (conversaciones inventadas)")
    cols = st.columns(len(PRESETS))
    for col, (name, p) in zip(cols, PRESETS.items()):
        col.button(p["label"], key=f"preset_{name}", on_click=load_preset, args=(name,), width="stretch")

    c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 1.2])
    channel = c1.radio("Canal", ["whatsapp", "instagram"], horizontal=True, format_func=channel_name, key="chat_channel")
    dow = c2.selectbox("Día", range(1, 8), format_func=lambda d: DAYS[d - 1].capitalize(), key="chat_dow")
    prior = c3.number_input("¿Cuántas veces ha escrito antes?", 0, 200, key="chat_prior")
    st.session_state.chat_prior_h = min(st.session_state.get("chat_prior_h", 0), int(prior))
    prior_h = c4.number_input("¿Cuántas de esas necesitaron persona?", 0, int(prior), key="chat_prior_h",
                              disabled=prior == 0)

    st.markdown("**Los primeros 4 mensajes**, en orden. La hora del primero es la hora de inicio.")
    day = datetime(2026, 10, 26) + timedelta(days=dow - 1)  # a Monday, shifted to the chosen weekday
    messages, problems, prev = [], [], None
    for i in range(4):
        de = st.session_state.get(f"b{i}_de", "cliente")
        with st.chat_message("user" if de == "cliente" else "assistant",
                             avatar=":material/person:" if de == "cliente" else ":material/smart_toy:"):
            a, b, c = st.columns([1.6, 0.8, 1])
            a.radio("Quién lo envía", list(SENDERS), format_func=SENDERS.get, horizontal=True, key=f"b{i}_de")
            hora = b.text_input("Hora", key=f"b{i}_hora", placeholder="10:15:00")
            adj = c.selectbox("Adjunto", list(ATTACHMENTS), format_func=ATTACHMENTS.get, key=f"b{i}_adj")
            texto = st.text_area("Texto", key=f"b{i}_texto", height=68, label_visibility="collapsed",
                                 placeholder="Escribe el mensaje")
        t = parse_time(hora or "")
        if t is None:
            problems.append(f"Mensaje {i + 1}: la hora debe tener la forma 10:15 o 10:15:30.")
            continue
        ts = datetime.combine(day, t)
        while prev is not None and ts < prev:  # past midnight
            ts += timedelta(days=1)
        prev = ts
        if not texto.strip() and not adj:
            problems.append(f"Mensaje {i + 1}: escribe un texto o elige un adjunto.")
        messages.append({"de": st.session_state[f"b{i}_de"], "texto": texto, "ts": ts, "adjunto": adj or None})
    if not any(m["de"] == "cliente" for m in messages):
        problems.append("Al menos uno de los 4 mensajes debe ser del cliente.")
    if problems:
        for p in problems:
            st.warning(p)
        return None
    row = features_from_chat(messages, channel, prior, prior_h)
    return pd.DataFrame([{"empresa": "—", "cliente": "—", "fecha": pd.Timestamp(messages[0]["ts"].date()), **row}])


def computed_table(row):
    r = row.iloc[0]

    def secs(v):
        return "—" if pd.isna(v) else f"{v:.0f} s"

    items = [
        ("Hora de inicio y día", f"{DAYS[r.dow - 1]} {r.hour_local}:00"),
        ("Mensajes del cliente / de la atención automática", f"{r.w_customer_msgs} / {r.w_ai_msgs}"),
        ("Largo promedio de los mensajes del cliente", f"{r.cust_chars_mean:.1f} caracteres"),
        ("Largo del mensaje más largo", f"{r.cust_chars_max} caracteres"),
        ("Signos «?» del cliente", r.cust_question_marks),
        ("Mensajes repetidos", r.cust_repeats),
        ("Fotos · audios · documentos · otros", f"{r.media_image} · {r.media_audio} · {r.media_document} · {r.media_other}"),
        ("Segundos hasta la primera respuesta automática (negativo: escribió primero)", secs(r.first_ai_reply_s)),
        ("Segundos promedio entre mensajes del cliente", secs(r.cust_gap_mean_s)),
        ("La atención automática dijo que no tenía la información", "sí" if r.ai_no_info else "no"),
        ("El cliente pidió un asesor", "sí" if r.cust_asks_human else "no"),
        ("El cliente usó palabras de queja", "sí" if r.cust_negative_words else "no"),
        ("Conversaciones anteriores / que necesitaron persona", f"{r.prior_conversations} / {r.prior_needed_human}"),
    ]
    return pd.DataFrame(items, columns=["Qué se calculó", "Valor"]).astype(str)


def manual_row():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Contexto**")
        channel = st.radio("Canal", ["whatsapp", "instagram"], horizontal=True, format_func=channel_name, key="m_channel")
        dow = st.selectbox("Día", range(1, 8), format_func=lambda d: DAYS[d - 1].capitalize(), key="m_dow")
        hour = st.slider("Hora (Colombia)", 0, 23, 10, key="m_hour")
        prior = st.number_input("Conversaciones anteriores del cliente", 0, 200, 0, key="m_prior")
        prior_h = st.number_input("De esas, cuántas necesitaron persona", 0, int(prior), 0, key="m_prior_h") if prior else 0
    with c2:
        st.markdown("**Mensajes del cliente (primeros 4)**")
        n_cust = st.slider("Mensajes del cliente", 1, 4, 2, key="m_ncust")
        chars_mean = st.number_input("Largo promedio (caracteres)", 1, 3000, 25, key="m_mean")
        chars_max = st.number_input("Largo del más largo", int(chars_mean), 3000, max(int(chars_mean), 34), key="m_max")
        qmarks = st.number_input("Signos «?»", 0, 20, 0, key="m_q")
        repeats = st.number_input("Mensajes repetidos", 0, 3, 0, key="m_rep")
        gap = st.number_input("Segundos promedio entre sus mensajes", 0, 10 ** 7, 34, disabled=n_cust == 1, key="m_gap")
        m1, m2 = st.columns(2)
        img = m1.number_input("Fotos", 0, 4, 0, key="m_img"); aud = m2.number_input("Audios", 0, 4, 0, key="m_aud")
        doc = m1.number_input("Documentos", 0, 4, 0, key="m_doc"); oth = m2.number_input("Otros adjuntos", 0, 4, 0, key="m_oth")
    with c3:
        st.markdown("**Atención automática**")
        replied = st.checkbox("Respondió dentro de los primeros 4 mensajes", value=n_cust < 4, disabled=n_cust < 4, key="m_replied")
        first = st.number_input("Segundos hasta la primera respuesta", -600, 10 ** 6, 12, disabled=not replied, key="m_first")
        no_info = st.checkbox("Dijo que no tenía la información", key="m_noinfo")
        st.markdown("**Señales de texto**")
        asks = st.checkbox("El cliente pidió un asesor", key="m_asks")
        neg = st.checkbox("El cliente usó palabras de queja", key="m_neg")
        st.caption("Aparecen en menos del 1 % de las conversaciones, así que el modelo casi no las usa. "
                   "Un pedido explícito de asesor se atiende con una regla, no con el modelo.")
    replied = replied or n_cust < 4
    return pd.DataFrame([{
        "empresa": "—", "cliente": "—", "fecha": pd.Timestamp.today().normalize(), "channel": channel,
        "hour_local": hour, "dow": dow, "w_customer_msgs": n_cust, "w_ai_msgs": 4 - n_cust,
        "cust_chars_mean": float(chars_mean), "cust_chars_max": int(chars_max), "cust_question_marks": qmarks,
        "cust_repeats": repeats, "media_image": img, "media_audio": aud, "media_document": doc, "media_other": oth,
        "first_ai_reply_s": float(first) if replied else np.nan, "cust_gap_mean_s": float(gap) if n_cust > 1 else np.nan,
        "ai_chars_mean": np.nan, "ai_no_info": int(no_info), "cust_asks_human": int(asks),
        "cust_negative_words": int(neg), "prior_conversations": prior, "prior_needed_human": prior_h,
    }])


def tab_chat():
    st.caption("Escribe o carga una conversación tal como la vería el equipo. La app calcula sola las variables "
               "que usa el modelo, con las mismas reglas con que se construyó el conjunto de datos.")
    row = chat_editor()
    if row is not None:
        st.divider()
        show_result(row)
        with st.expander("Lo que la app calculó de la conversación"):
            st.dataframe(computed_table(row), hide_index=True, width="stretch")
    with st.expander("Modo avanzado: ingresar las variables a mano"):
        adv = manual_row()
        st.divider()
        show_result(adv)


# --- Inbox ---------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Calculando el riesgo de cada conversación…")
def rank(raw):
    model = get_model()
    out = raw.reset_index(drop=True).copy()
    out = out.join(model.risk(out)[["score", "nivel"]]).sort_values("score", ascending=False).reset_index(drop=True)
    top = out.head(REASONS_LIMIT)
    sv, _ = model.shap_values(top)
    main = []
    for i in top.index:
        reasons = explain(top.loc[i], sv.loc[i], top=3)
        # A red or yellow row is explained by what pushes it up; a green one by what pulls it down.
        want = "baja" if out.at[i, "nivel"] == "verde" else "sube"
        pick = next((r for r in reasons if r[2] == want), reasons[0] if reasons else None)
        main.append(f"{'▲' if pick[2] == 'sube' else '▼'} {pick[0]}" if pick else "—")
    out["razon_principal"] = main + ["—"] * (len(out) - len(main))
    return out


def level_style(v):
    for color, name, _ in LEVELS.values():
        if v == name:
            return f"background-color: {color}33; font-weight: 600"
    return ""


def inbox_table(ranked, key):
    view = pd.DataFrame({
        "Nivel": ranked.nivel.map(lambda n: LEVELS[n][1]),
        "Riesgo": (100 * ranked.score).round(0),
        "Razón principal": ranked.razon_principal,
        "Canal": ranked.channel.map(channel_name),
        "Día y hora": [f"{DAYS[int(d) - 1]} {int(h)}:00" for d, h in zip(ranked.dow, ranked.hour_local)],
    })
    if "empresa" in ranked:
        view.insert(3, "Empresa", ranked.empresa)
    return st.dataframe(
        view.style.map(level_style, subset=["Nivel"]), hide_index=True, width="stretch", key=key,
        on_select="rerun", selection_mode="single-row", height=min(38 + 35 * len(view), 460),
        column_config={"Riesgo": st.column_config.ProgressColumn("Riesgo", format="%d %%", min_value=0, max_value=100),
                       "Razón principal": st.column_config.TextColumn(width="large")})


def tab_inbox():
    st.caption("La cola de conversaciones ordenada de mayor a menor riesgo, para atender primero las que más "
               "probablemente van a necesitar a una persona. Selecciona una fila para ver el detalle.")
    source = st.radio("Conversaciones", ["Ejemplos de octubre", "Subir un CSV"], horizontal=True, key="inbox_source")
    if source == "Ejemplos de octubre":
        raw = get_examples().drop(columns="etiqueta")
    else:
        st.markdown(f"El CSV necesita las columnas del conjunto de datos (diccionario en "
                    f"`docs/diccionario_de_datos.md`): `{'`, `'.join(RAW_FIELDS)}`. `empresa`, `cliente`, `fecha` y "
                    f"`{TARGET}` son opcionales.")
        sample = get_examples().drop(columns=["etiqueta", TARGET, "mes"])
        st.download_button("Descargar un CSV de ejemplo", sample.to_csv(index=False).encode(), "bandeja_ejemplo.csv",
                           "text/csv")
        up = st.file_uploader("CSV con las conversaciones", type="csv", key="inbox_csv")
        if up is None:
            return
        try:
            raw = pd.read_csv(up)
        except Exception as e:  # malformed upload: tell the user instead of crashing the page
            st.error(f"No se pudo leer el archivo: {e}")
            return
        missing = [c for c in RAW_FIELDS if c not in raw.columns]
        if missing:
            st.error(f"Faltan columnas: {', '.join(missing)}.")
            return
        if raw.empty:
            st.warning("El archivo no tiene filas.")
            return
    ranked = rank(raw)
    counts = ranked.nivel.value_counts()
    c = st.columns(4)
    c[0].metric("Conversaciones", len(ranked))
    for col, n in zip(c[1:], LEVELS):
        col.metric(LEVELS[n][1], int(counts.get(n, 0)))
    if len(ranked) > REASONS_LIMIT:
        st.caption(f"La razón principal se calcula para las {REASONS_LIMIT} de mayor riesgo.")
    sel = inbox_table(ranked, key=f"inbox_{source}")
    export = ranked.drop(columns=["mes"], errors="ignore").rename(columns={"score": "riesgo"})
    st.download_button("Descargar el ranking (CSV)", export.to_csv(index=False).encode(), "bandeja_ranking.csv", "text/csv")
    rows = sel.selection.rows
    if rows:
        st.divider()
        pick = ranked.iloc[[rows[0]]]
        st.subheader(f"Conversación n.º {rows[0] + 1} de la cola")
        truth = pick[TARGET].iloc[0] if TARGET in pick and pd.notna(pick[TARGET].iloc[0]) else None
        show_result(pick.drop(columns=["score", "nivel", "razon_principal"]),
                    truth=None if truth is None else int(truth))


# --- Examples and model explanation --------------------------------------------------------------

def tab_examples():
    st.info("**Ejemplos ilustrativos.** El modelo que usa la app se reentrenó con todos los meses, incluido "
            "octubre, así que ya vio estas conversaciones. La evaluación sin ver los datos (entrenado hasta "
            "septiembre y probado en octubre) está en el notebook `model_development.ipynb` y en la pestaña "
            "«Cómo funciona el modelo».", icon=":material/info:")
    ex = get_examples()
    choice = st.selectbox("Conversación de octubre", ex.etiqueta)
    row = ex[ex.etiqueta == choice].drop(columns="etiqueta")
    show_result(row, truth=int(row[TARGET].iloc[0]))
    with st.expander("Datos de la conversación"):
        st.dataframe(row.drop(columns=[TARGET, "mes"]).T.rename(columns=lambda _: "valor").astype(str), width="stretch")


def card(col, big, label, detail):
    with col.container(border=True):
        st.markdown(f"<div style='font-size:1.6rem;font-weight:700;line-height:1.2'>{big}</div>"
                    f"<div style='font-weight:600;margin:2px 0 6px'>{label}</div>"
                    f"<div style='opacity:.7;font-size:.85rem'>{detail}</div>", unsafe_allow_html=True)


def es(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def miles(n):
    return f"{n:,}".replace(",", ".")


def tab_model():
    m, model = get_metrics(), get_model()
    o, biz, new = m["octubre"]["Octubre"], m["negocio_octubre"], m["octubre"]["Octubre, clientes nuevos"]
    meta = m["meta_final"]
    st.subheader("Qué tan bien funciona")
    st.caption("Prueba honesta: el modelo se entrenó con las conversaciones hasta el 30 de septiembre de 2026 y "
               "se evaluó en las de octubre, que nunca vio.")
    c = st.columns(4)
    card(c[0], f"{round(biz['Alertas acertadas por cada 10'])} de cada 10", "alertas son correctas",
         f"Precisión {es(o['precisión'])}. Si se alertara al azar serían {es(biz['Acertadas por cada 10 si se alertara al azar'], 1)}.")
    card(c[1], f"{round(10 * o['recall'])} de cada 10", "casos que necesitaron persona se detectan",
         f"Recall {es(o['recall'])}: {miles(biz['Detectadas al mensaje 4'])} de {miles(biz['Necesitaron persona'])} al cuarto mensaje.")
    card(c[2], f"{biz['Alertas por cada 100 conversaciones']:.0f} de cada 100", "conversaciones llegan en rojo",
         "Es la carga de trabajo que genera el modelo para el equipo.")
    card(c[3], es(o["PR-AUC"]), "calidad del orden de la cola (PR-AUC)",
         f"Al azar sería {es(o['azar PR'])}. ROC-AUC {es(o['ROC-AUC'])}; con clientes nuevos {es(new['ROC-AUC'])}.")

    st.subheader("Con qué datos")
    n = meta["n"]
    scope = pd.DataFrame([
        ("Empresas", f"{len(meta['empresas'])}: {meta['empresas'][0]}–E08 y E10–{meta['empresas'][-1]}. "
                     f"Se excluye {', '.join(meta['excluye'])} (92 % de los datos, operación mixta)."),
        ("Periodo", f"Del {meta['fecha_min']} al {meta['fecha_max']}; el 88 % entre julio y octubre de 2026."),
        ("Conversaciones", f"{miles(n)} de 3.507 clientes, con {miles(4 * n)} mensajes (4 por conversación)."),
        ("Partición", "Entrenamiento hasta el 30 de septiembre de 2026 (2.806); prueba en octubre (2.465)."),
        ("Modelo final", f"Random Forest calibrado, reentrenado con las {miles(n)} conversaciones. Alerta desde "
                         f"{model.thresholds['alerta']:.0%}, vigilar desde {model.thresholds['vigilar']:.0%}."),
    ], columns=["", "Detalle"])
    st.dataframe(scope, hide_index=True, width="stretch")

    st.subheader("Cuánto se adelanta")
    c = st.columns(3)
    card(c[0], ANTICIPATION["antes"], "de las alertas llegan antes de que escriba una persona",
         "La predicción se hace al cuarto mensaje, antes de cualquier intervención humana.")
    card(c[1], ANTICIPATION["mediana"], "de margen típico en los casos en rojo",
         "Mediana de minutos entre la alerta y el primer mensaje de una persona, octubre.")
    card(c[2], ANTICIPATION["5min"], "alertas en rojo dejan 5 minutos o más",
         "Medido en producción con una consulta de solo lectura (documento técnico §3.7).")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Qué mira el modelo")
        st.image(str(ROOT / "figures" / "14_shap_beeswarm.png"), width="stretch")
        st.caption("Cada punto es una conversación de octubre. A la derecha del cero, la variable sube el riesgo; "
                   "a la izquierda, lo baja. Lo que más pesa: la proporción de conversaciones anteriores que "
                   "necesitaron persona (prior_human_rate, prior_needed_human), el tiempo de la primera respuesta "
                   "automática (log_first_ai_reply_s), el largo de los mensajes del cliente y el canal Instagram.")
    with right:
        st.subheader("Rendimiento por mes")
        st.image(str(ROOT / "figures" / "21_seguimiento_mensual.png"), width="stretch")
        st.caption("Cada mes, evaluado con el modelo entrenado hasta el mes anterior. Septiembre bajó porque "
                   "llegaron 3 empresas que el modelo no conocía; en octubre se recupera el PR-AUC, pero el "
                   "recall queda en 0,58–0,62 porque cambió la mezcla de empresas.")

    st.subheader("Cuándo se reentrena")
    st.markdown(
        "- **Cada día, sin esperar etiquetas:** se compara la distribución de cada variable y del puntaje con la de "
        "entrenamiento (PSI). Desde 0,25 es un cambio fuerte.\n"
        "- **Cada mes:** recall, precisión y calibración por empresa. Se reentrena si el recall cae más de 5 puntos, "
        "llegan empresas nuevas, alguna variable pasa de PSI 0,25, la calibración se desvía más de 5 puntos, cambia el "
        "modelo de IA de la plataforma o pasan 3 meses.\n"
        "- **Para reemplazar al modelo vigente,** el nuevo debe ganar en PR-AUC y no perder más de 2 puntos de recall "
        "alertando el mismo volumen de conversaciones. Luego corre en paralelo una o dos semanas y el anterior se "
        "conserva para volver atrás.\n"
        "- **Con una empresa nueva,** el modelo solo ordena la cola hasta tener un mes de datos suyos.")
    st.caption("Las razones que muestra la app son correlaciones aprendidas de datos históricos, no causas.")


def sidebar():
    model, m = get_model(), get_metrics()
    o = m["octubre"]["Octubre"]
    st.sidebar.header("Modelo")
    st.sidebar.write(f"**{model.meta.get('version', '')}**: Random Forest calibrado, sin `ai_chars_mean`.")
    st.sidebar.write(f"Entrenado con {model.meta['n']:,} conversaciones ({model.meta['fecha_min']} a "
                     f"{model.meta['fecha_max']}), {len(model.meta['empresas'])} empresas.".replace(",", "."))
    st.sidebar.caption(f"Empresas {', '.join(model.meta['empresas'])}. Excluye {', '.join(model.meta.get('excluye', []))} "
                       "(92 % de los datos, operación mixta). 3.507 clientes; solo conversaciones con 4 mensajes "
                       "de atención automática sin intervención humana. La prueba de octubre usó un modelo "
                       "entrenado hasta el 30 de septiembre de 2026 (2.806 conversaciones).")
    st.sidebar.subheader("Prueba en octubre (no vista)")
    st.sidebar.markdown(f"- ROC-AUC **{o['ROC-AUC']:.2f}**, PR-AUC **{o['PR-AUC']:.2f}** (azar {o['azar PR']:.2f})\n"
                        f"- Detecta **{o['recall']:.0%}** de los casos con **{o['precisión']:.0%}** de alertas acertadas")
    st.sidebar.subheader("Semáforo")
    st.sidebar.markdown(f"- **Alerta** desde {model.thresholds['alerta']:.0%}\n- **Vigilar** desde {model.thresholds['vigilar']:.0%}")
    st.sidebar.caption("Las razones son correlaciones aprendidas de datos históricos, no causas.")


st.title("Detección temprana de conversaciones que van a necesitar una persona")
st.caption("Con los primeros 4 mensajes de una conversación que empezó la atención automática, antes de que "
           "intervenga alguien del equipo.")
sidebar()
t_chat, t_inbox, t_ex, t_model = st.tabs(["Conversación", "Bandeja", "Ejemplos reales (anonimizados)",
                                          "Cómo funciona el modelo"])
with t_chat:
    tab_chat()
with t_inbox:
    tab_inbox()
with t_ex:
    tab_examples()
with t_model:
    tab_model()
