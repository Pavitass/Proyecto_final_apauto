# Detección temprana de conversaciones en riesgo

> Proyecto final de Machine Learning · Clasificación binaria supervisada · Python + scikit-learn

Con los **primeros 4 mensajes** de una conversación que empezó con atención automática (agente de IA
o flujo), y **antes de cualquier intervención humana**, el modelo predice si esa conversación va a
necesitar a una persona del equipo. El resultado se presenta como un semáforo (rojo / amarillo /
verde) con razones en lenguaje simple.

Los datos son conversaciones reales y anonimizadas de **Orvyz** (WhatsApp e Instagram), de julio a
octubre de 2026.

**Resultado principal:** entrenado con datos hasta septiembre y probado con octubre (un mes que el
modelo nunca vio, sin la empresa dominante), alcanza **ROC-AUC 0,79 y PR-AUC 0,77**, frente a 0,44 al
azar. Detalle completo en [`model_card.md`](model_card.md).

- **Repositorio:** <https://github.com/Pavitass/Proyecto_final_apauto>
- **Video de demostración (2 min):** _pendiente de publicar_ · se agregará el enlace aquí
- **Interfaz en vivo:** _pendiente de publicar_ (Streamlit Community Cloud)

---

## Tabla de contenido

- [Qué problema resuelve](#qué-problema-resuelve)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Cómo ejecutarlo](#cómo-ejecutarlo)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Resultados](#resultados-destacados)
- [Correspondencia con el enunciado](#correspondencia-con-el-enunciado)

---

## Qué problema resuelve

En Orvyz, una plataforma SaaS de atención al cliente, un agente de IA o un flujo automático responde
primero. Cuando no alcanza, interviene una persona del equipo, a menudo tarde, cuando el cliente ya
repitió la pregunta o se frustró. Este modelo anticipa ese momento para:

1. Pasar el chat a una persona **antes** de que el cliente se frustre (alerta roja).
2. Ordenar la cola de atención por riesgo.
3. Detectar qué agentes automáticos escalan más de lo esperado y necesitan ajustes.

El paradigma es **aprendizaje supervisado, clasificación binaria**. La etiqueta (`needed_human`) se
genera sola al cerrar cada conversación, así que no hace falta etiquetar a mano.

## Requisitos

- **Python 3.10 o superior**
- Las dependencias de [`requirements.txt`](requirements.txt): pandas, numpy, scikit-learn,
  matplotlib, seaborn, shap, jupyter, nbconvert, ipykernel, streamlit, joblib.

## Instalación

Clonar el repositorio y crear un entorno virtual con las dependencias.

<details open>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Linux / macOS (bash)</strong></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

> Tras activar el entorno, los comandos de abajo pueden escribirse directamente (`python`,
> `jupyter`, `streamlit`). Si prefieres no activarlo, usa la ruta del ejecutable del entorno:
> `.venv\Scripts\python` en Windows o `.venv/bin/python` en Linux/macOS.

## Cómo ejecutarlo

### 1. Reproducir el notebook entregable

Regenera los artefactos de `models/` y las gráficas de `figures/`.

<details open>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace model_development.ipynb
```
</details>

<details>
<summary><strong>Linux / macOS (bash)</strong></summary>

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace model_development.ipynb
```
</details>

Los notebooks de [`notebooks/`](notebooks/) (01 → 02 → 03) recorren los mismos pasos por partes.

### 2. Abrir la interfaz

<details open>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
.venv\Scripts\streamlit run app/app.py
```
</details>

<details>
<summary><strong>Linux / macOS (bash)</strong></summary>

```bash
.venv/bin/streamlit run app/app.py
```
</details>

### 3. Reentrenar y comparar contra el modelo vigente

<details open>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
.venv\Scripts\python src/retrain.py --data data/orvyz_conversaciones_riesgo.csv --champion models/riesgo_v1.joblib --holdout 2026-10 --out models/riesgo_v2.joblib
```
</details>

<details>
<summary><strong>Linux / macOS (bash)</strong></summary>

```bash
.venv/bin/python src/retrain.py --data data/orvyz_conversaciones_riesgo.csv --champion models/riesgo_v1.joblib --holdout 2026-10 --out models/riesgo_v2.joblib
```
</details>

El retador reemplaza al vigente solo si gana en PR-AUC y no pierde más de 2 puntos de recall
alertando el mismo volumen de conversaciones.

## Estructura del repositorio

```
model_development.ipynb   Notebook entregable: EDA, preprocesamiento, modelado, fuga, métricas,
                          SHAP y reajuste. Es la fuente de todas las cifras.
model_card.md             Ficha del modelo: métricas, umbrales, limitaciones y cómo calcular cada
                          entrada desde los primeros 4 mensajes.
requirements.txt          Dependencias de Python.

data/                     orvyz_conversaciones_riesgo.csv (anonimizado) y data/timing/ (resultados
                          de sql/timing.sql).
notebooks/                Los mismos pasos en tres partes: 01 EDA y preprocesamiento ·
                          02 modelado, calibración y SHAP · 03 monitoreo y reentrenamiento.
src/                      features.py (variables) · risk_model.py (modelo, calibración, explicación)
                          monitoring.py (PSI, métricas por empresa, regla de reentrenamiento) ·
                          retrain.py (retador vs. vigente) · baseline.py (prueba inicial) ·
                          export_production.py (export JSON) · timing.py (anticipación).
app/                      app.py (interfaz Streamlit para usuarios no técnicos).
models/                   riesgo_v1.joblib · best_params.json · metrics.json · export de producción.
figures/                  Gráficas exportadas por los notebooks.
docs/                     documento_tecnico.md (documento del proyecto, 3.1–3.9 y matriz de
                          cumplimiento) · propuesta · diccionario de datos · resumen del equipo.
sql/                      extract.sql (extracción original, solo lectura) · timing.sql
                          (escalaciones y anticipación, solo lectura).
```

## Resultados destacados

Prueba temporal: entrenado con julio–septiembre de 2026, probado con octubre (2.465 conversaciones
nunca vistas, sin la empresa dominante).

| Métrica | Octubre | Solo clientes nuevos |
|---|---|---|
| ROC-AUC | 0,79 | 0,74 |
| PR-AUC (azar entre paréntesis) | 0,77 (0,44) | 0,78 (0,54) |
| Recall con umbral de alerta | 0,62 | 0,66 |
| Precisión con umbral de alerta | 0,71 | 0,70 |

En octubre se detectaron **6 de cada 10 casos** al cuarto mensaje y **7 de cada 10 alertas fueron
correctas** (contra 4,4 al azar). En el 10 % más riesgoso, el 96 % necesitó persona. Más detalle y
el análisis de generalización, fuga y calibración están en el notebook y en `model_card.md`.

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
