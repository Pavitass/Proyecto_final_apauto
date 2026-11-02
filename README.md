# Proyecto: Detección Temprana de Conversaciones en Riesgo

Con los primeros 4 mensajes de una conversación que empezó la atención automática (agente de IA o
flujo), y antes de cualquier intervención humana, se predice si la conversación va a necesitar a
una persona del equipo. Los datos son conversaciones reales y anonimizadas de Orvyz (WhatsApp e
Instagram) de 11 empresas, del 20 de julio de 2025 al 29 de octubre de 2026.

Resultado principal: entrenado con datos hasta septiembre y probado con octubre (sin la empresa
dominante), **ROC-AUC 0,79 y PR-AUC 0,77, contra 0,44 al azar**. Detalle en
[`model_card.md`](model_card.md).

## Alcance de los datos

El modelo no cubre todos los clientes de Orvyz: se construyó con empresas y un periodo específicos.

| | |
|---|---|
| **Extracción original** | 62.788 conversaciones de 12 empresas y 4.347 clientes, del 20 de julio de 2025 al 29 de octubre de 2026. |
| **Empresas** | 11 de 12: **E01–E08 y E10–E12**. **E09 queda fuera**: es el 92 % de las conversaciones y necesita persona en el 95 % de los casos porque su operación es mixta por diseño; con ella adentro, el modelo aprendería la operación de E09 y no el problema. |
| **Conjunto del modelo** | **5.271 conversaciones** de **3.507 clientes**. Solo entran conversaciones que empezó la atención automática y que llegaron a 4 mensajes (del cliente y de la IA o el flujo) sin que interviniera una persona: **21.084 mensajes** en la ventana (5.271 × 4). |
| **Periodo** | Del 20 de julio de 2025 al 29 de octubre de 2026. El **88 %** está entre julio y octubre de 2026 (septiembre 1.861, octubre 2.465); antes casi no había atención automática. |
| **Partición** | Entrenamiento: todo hasta el **30 de septiembre de 2026** (2.806 conversaciones). Prueba: **octubre de 2026** (2.465), un mes que el modelo no vio. El modelo final (`riesgo_v1`) se reentrena con las 5.271. |

Los resultados valen para estas empresas y este periodo; con otras empresas o en otro momento hay
que volver a validarlo (estrategia de reajuste, §3.9 del documento técnico).

## Estructura

```
model_development.ipynb  notebook entregable: EDA, preprocesamiento, modelado, fuga, métricas, SHAP y reajuste
data/        orvyz_conversaciones_riesgo.csv (anonimizado)
docs/        documento_tecnico.md (documento del proyecto, 3.1–3.9 y matriz de cumplimiento) · propuesta · diccionario de datos
notebooks/   los mismos pasos en tres partes: 01 EDA y preprocesamiento · 02 modelado, calibración y SHAP · 03 monitoreo y reentrenamiento
src/         features.py (variables) · risk_model.py (modelo, calibración, explicación)
             monitoring.py (PSI, métricas por empresa, regla de reentrenamiento) · retrain.py (retador vs vigente)
             baseline.py (prueba inicial)
app/         app.py (interfaz Streamlit)
models/      riesgo_v1.joblib · best_params.json · metrics.json
figures/     gráficas exportadas por los notebooks
sql/         extract.sql (extracción original, solo lectura) · timing.sql (escalaciones en la ventana y anticipación en minutos, solo lectura; resultados en data/timing/)
model_card.md  ficha del modelo: métricas, umbrales, limitaciones y cómo calcular cada entrada
```

## Cómo correrlo

Instalar dependencias:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Reproducir el notebook entregable (regenera `models/` y `figures/`):

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace model_development.ipynb
```

Los notebooks de `notebooks/` (01 → 02 → 03) son el mismo recorrido por partes.

Abrir la interfaz:

```bash
.venv/bin/streamlit run app/app.py
```

Reentrenar y comparar contra el modelo vigente:

```bash
.venv/bin/python src/retrain.py --data data/orvyz_conversaciones_riesgo.csv --champion models/riesgo_v1.joblib --holdout 2026-10 --out models/riesgo_v2.joblib
```

## Correspondencia con el enunciado

El documento completo, con la matriz de cumplimiento y los pendientes, está en
[`docs/documento_tecnico.md`](docs/documento_tecnico.md). En `model_development.ipynb`:

| Punto | Dónde |
|---|---|
| 3.1 Problema | introducción del notebook; `docs/documento_tecnico.md` §3.1 |
| 3.2 Preprocesamiento | Parte 1 §1, §8; `src/features.py` |
| 3.3 EDA | Parte 1 |
| 3.4 Paradigma | supervisado, clasificación binaria |
| 3.5 Modelos y selección | Parte 2 §1–2 |
| 3.6 Generalización | Parte 2 §1–3 y §3.b (auditoría de fuga) |
| 3.7 Métricas | Parte 2 §4–5 (PR-AUC, recall/precisión, calibración, métricas de negocio) |
| Explicabilidad | Parte 2 §6 (SHAP global, huella de empresa, por predicción) |
| 3.8 Despliegue e interfaz | `app/app.py`, `models/`, `model_card.md` |
| 3.9 Reentrenamiento | Parte 3; `src/monitoring.py`, `src/retrain.py` |
