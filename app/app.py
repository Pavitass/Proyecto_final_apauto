"""Interfaz: riesgo de que una conversación necesite a una persona, con semáforo y razones.

    .venv/bin/streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import DOMINANT, TARGET, load  # noqa: E402
from risk_model import DAYS, explain  # noqa: E402

LEVELS = {
    "rojo": ("#e34948", "Alerta", "Pasar a una persona del equipo ahora."),
    "amarillo": ("#eda100", "Vigilar", "Dejar con la atención automática, pero ponerla arriba en la cola."),
    "verde": ("#1baf7a", "Bajo", "La atención automática probablemente la resuelve sola."),
}

st.set_page_config(page_title="Riesgo de conversación", layout="wide")


@st.cache_resource
def get_model():
    return joblib.load(ROOT / "models" / "riesgo_v1.joblib")


@st.cache_data
def get_metrics():
    return json.load(open(ROOT / "models" / "metrics.json"))


@st.cache_data
def get_examples(per_level=8):
    df = load(ROOT / "data" / "orvyz_conversaciones_riesgo.csv")
    oct_ = df[(df.empresa != DOMINANT) & (df.mes == "2026-10")]
    lv = get_model().risk(oct_)
    picks = [oct_.loc[lv.index[lv.nivel == n]].sample(per_level, random_state=1) for n in LEVELS]
    ex = pd.concat(picks).sample(frac=1, random_state=2).reset_index(drop=True)
    ex["etiqueta"] = [f"Ejemplo {i + 1:02d} · {r.empresa} · {'Instagram' if r.channel == 'instagram' else 'WhatsApp'} · "
                      f"{DAYS[r.dow - 1]} {r.hour_local}:00" for i, r in ex.iterrows()]
    return ex


def badge(level):
    color, name, action = LEVELS[level]
    return (f"<div style='display:flex;align-items:center;gap:12px;padding:14px 16px;border-radius:12px;"
            f"border:1px solid {color};background:{color}1a'>"
            f"<div style='width:22px;height:22px;border-radius:50%;background:{color};flex:none'></div>"
            f"<div><div style='font-weight:600;font-size:1.05rem'>{name}</div>"
            f"<div style='opacity:.8;font-size:.9rem'>{action}</div></div></div>")


def risk_bar(score, th):
    marks = "".join(f"<div title='{n}' style='position:absolute;left:{100 * v:.1f}%;top:-4px;bottom:-4px;width:2px;"
                    f"background:currentColor;opacity:.55'></div>" for n, v in th.items())
    return (f"<div style='position:relative;height:10px;border-radius:5px;background:rgba(128,128,128,.2);margin:10px 0 4px'>"
            f"<div style='width:{100 * score:.1f}%;height:100%;border-radius:5px;background:{LEVELS[level_of(score, th)][0]}'></div>"
            f"{marks}</div><div style='font-size:.75rem;opacity:.7'>"
            f"Marcas: vigilar {th['vigilar']:.0%} · alerta {th['alerta']:.0%}</div>")


def level_of(score, th):
    return "rojo" if score >= th["alerta"] else "amarillo" if score >= th["vigilar"] else "verde"


def show_result(row, truth=None):
    model = get_model()
    r = model.risk(row).iloc[0]
    sv, _ = model.shap_values(row)
    reasons = explain(row.iloc[0], sv.iloc[0], top=5)
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.metric("Riesgo de necesitar una persona", f"{r.riesgo:.0%}")
        st.markdown(risk_bar(r.score, model.thresholds), unsafe_allow_html=True)
        st.markdown(badge(r.nivel), unsafe_allow_html=True)
        if truth is not None:
            with st.expander("¿Qué pasó en realidad?"):
                st.write("**Necesitó a una persona.**" if truth else "**La atención automática la resolvió sola.**")
    with right:
        st.markdown("**Por qué**")
        if not reasons:
            st.write("Ninguna señal se aleja del promedio.")
        for text, effect, direction in reasons:
            pts = round(100 * abs(effect))
            arrow, color = ("▲", "#e34948") if direction == "sube" else ("▼", "#1baf7a")
            st.markdown(f"<div style='padding:6px 0;border-bottom:1px solid rgba(128,128,128,.2)'>"
                        f"<span style='color:{color};font-weight:700'>{arrow}</span> {text} "
                        f"<span style='opacity:.65'>· {direction} {pts} punto{'' if pts == 1 else 's'}</span></div>",
                        unsafe_allow_html=True)
        st.caption("Cada razón agrupa variables relacionadas (valores SHAP). Los puntos son cuánto mueve la "
                   "razón la probabilidad respecto a una conversación promedio.")


def manual_row():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Contexto**")
        channel = st.radio("Canal", ["whatsapp", "instagram"], horizontal=True,
                           format_func=lambda c: "WhatsApp" if c == "whatsapp" else "Instagram")
        dow = st.selectbox("Día", range(1, 8), format_func=lambda d: DAYS[d - 1].capitalize())
        hour = st.slider("Hora (Colombia)", 0, 23, 10)
        prior = st.number_input("Conversaciones anteriores del cliente", 0, 200, 0)
        prior_h = st.number_input("De esas, cuántas necesitaron persona", 0, int(prior), 0) if prior else 0
    with c2:
        st.markdown("**Mensajes del cliente (primeros 4)**")
        n_cust = st.slider("Mensajes del cliente", 1, 4, 2)
        chars_mean = st.number_input("Largo promedio (caracteres)", 1, 3000, 25)
        chars_max = st.number_input("Largo del más largo", int(chars_mean), 3000, max(int(chars_mean), 34))
        qmarks = st.number_input("Signos «?»", 0, 20, 0)
        repeats = st.number_input("Mensajes repetidos", 0, 3, 0)
        gap = st.number_input("Segundos promedio entre sus mensajes", 0, 10 ** 7, 34, disabled=n_cust == 1)
        m1, m2 = st.columns(2)
        img = m1.number_input("Fotos", 0, 4, 0); aud = m2.number_input("Audios", 0, 4, 0)
        doc = m1.number_input("Documentos", 0, 4, 0); oth = m2.number_input("Otros adjuntos", 0, 4, 0)
    with c3:
        st.markdown("**Atención automática**")
        replied = st.checkbox("Respondió dentro de los primeros 4 mensajes", value=n_cust < 4, disabled=n_cust < 4)
        first = st.number_input("Segundos hasta la primera respuesta", -600, 10 ** 6, 12, disabled=not replied)
        no_info = st.checkbox("Dijo que no tenía la información")
        st.markdown("**Señales de texto**")
        asks = st.checkbox("El cliente pidió un asesor")
        neg = st.checkbox("El cliente usó palabras de queja")
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
tab_ex, tab_manual = st.tabs(["Ejemplos reales (anonimizados)", "Ingresar una conversación"])
with tab_ex:
    ex = get_examples()
    choice = st.selectbox("Conversación de octubre", ex.etiqueta)
    row = ex[ex.etiqueta == choice].drop(columns="etiqueta")
    show_result(row, truth=int(row[TARGET].iloc[0]))
    with st.expander("Datos de la conversación"):
        st.dataframe(row.drop(columns=[TARGET, "mes"]).T.rename(columns=lambda _: "valor").astype(str), width="stretch")
    st.caption("Los ejemplos son de octubre. El modelo final se reentrenó con todos los meses, incluido octubre; "
               "la evaluación sin ver esos datos está en el notebook 02.")
with tab_manual:
    row = manual_row()
    st.divider()
    show_result(row)
