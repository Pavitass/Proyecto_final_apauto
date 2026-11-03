# Detección temprana de conversaciones en riesgo

> Proyecto final de Machine Learning · Clasificación binaria supervisada · Python + scikit-learn

Con los **primeros 4 mensajes** de una conversación que empezó con atención automática (agente de IA
o flujo), y **antes de cualquier intervención humana**, el modelo predice si esa conversación va a
necesitar a una persona del equipo. El resultado se presenta como un semáforo (rojo / amarillo /
verde) con razones en lenguaje simple.

Los datos son conversaciones reales y anonimizadas de **Orvyz** (WhatsApp e Instagram) de 11
empresas, del 20 de julio de 2025 al 29 de octubre de 2026.

**Resultado principal:** entrenado con todos los datos hasta el 30 de septiembre y probado con
octubre (un mes que el modelo nunca vio, sin la empresa dominante), alcanza **ROC-AUC 0,79 y
PR-AUC 0,77**, frente a 0,44 al azar. Detalle completo en [`model_card.md`](model_card.md).

- **Repositorio:** <https://github.com/Pavitass/Proyecto_final_apauto>
- **Video de demostración (2 min):** https://www.youtube.com/watch?v=odb7sy0w0uc
- **Interfaz en vivo:** <https://proyectofinalapauto-gos8hw7a3ademgukdhnjsh.streamlit.app/> (Streamlit Community Cloud)

---

## Tabla de contenido

- [Qué problema resuelve](#qué-problema-resuelve)
- [Alcance de los datos](#alcance-de-los-datos)
- [Variables del dataset](#variables-del-dataset)
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

## Variables del dataset

Cada fila de `data/orvyz_conversaciones_riesgo.csv` es una conversación. Salvo el historial y el
objetivo, todo sale de la **ventana**: los primeros 4 mensajes, todos anteriores a cualquier mensaje
de una persona del equipo. El CSV trae 25 columnas; `mes` la agrega `load()` al leerlo. Definición
exacta en [`sql/extract.sql`](sql/extract.sql), contrato de producción en
[`model_card.md`](model_card.md) y detalle en [`docs/diccionario_de_datos.md`](docs/diccionario_de_datos.md).

**Identificación y momento**

| Variable | Tipo | Significado |
|---|---|---|
| `empresa` | texto | Código anónimo de la empresa (E01–E12); no es entrada del modelo, se usa para validar por grupos. |
| `cliente` | texto | Código anónimo del chat (un cliente en un canal de una empresa); no es entrada del modelo. |
| `fecha` | fecha | Día de inicio de la conversación, en UTC. |
| `mes` | texto | Mes de `fecha` (`2026-10`), para la partición temporal; lo calcula `load()`. |
| `channel` | texto | Canal: `whatsapp` o `instagram`. |
| `hour_local` | entero | Hora de inicio en Colombia (0–23). |
| `dow` | entero | Día de la semana en Colombia, ISO: 1 = lunes … 7 = domingo (no hay 0). |

**Cliente (en la ventana)**

| Variable | Tipo | Significado |
|---|---|---|
| `w_customer_msgs` | entero | Cuántos de los 4 mensajes son del cliente (1–4). |
| `cust_chars_mean` / `cust_chars_max` | decimal / entero | Largo promedio y máximo, en caracteres, de los mensajes del cliente (un adjunto sin texto cuenta 0). |
| `cust_question_marks` | entero | Total de signos `?` que escribió el cliente. |
| `cust_repeats` | entero | Mensajes del cliente con texto idéntico a uno anterior suyo en la ventana, tras pasar a minúsculas y unir espacios. |
| `media_image` | entero | Adjuntos del cliente de tipo imagen. |
| `media_audio` | entero | Audios y notas de voz. |
| `media_document` | entero | Documentos (PDF y otros archivos). |
| `media_other` | entero | Cualquier otro adjunto, por ejemplo video o sticker. |
| `cust_gap_mean_s` | decimal | Segundos promedio entre mensajes consecutivos del cliente. |
| `cust_asks_human` | 0/1 | El cliente escribió `asesor`, `humano`, `una persona`, `agente real` o `hablar con alguien`. |
| `cust_negative_words` | 0/1 | El cliente escribió `queja`, `reclamo`, `pesimo`/`pésimo`, `terrible`, `molest…`, `estafa` o `nadie responde`. |

**IA o flujo automático (en la ventana)**

| Variable | Tipo | Significado |
|---|---|---|
| `w_ai_msgs` | entero | Cuántos de los 4 son automáticos; siempre `4 − w_customer_msgs`, por eso el modelo no la usa. |
| `first_ai_reply_s` | decimal | Segundos entre el primer mensaje del cliente y la primera respuesta automática; negativo si la IA escribió primero. |
| `ai_chars_mean` | decimal | Largo promedio de los mensajes automáticos; el modelo final no la usa porque delata a la empresa. |
| `ai_no_info` | 0/1 | Un mensaje automático dijo `no tengo`, `no cuento con`, `no dispongo`, `no puedo ayudarte`, `no tengo acceso` o `no encontr…`. |

**Historial y objetivo**

| Variable | Tipo | Significado |
|---|---|---|
| `prior_conversations` | entero | Conversaciones anteriores y ya cerradas del **mismo chat** (misma empresa y canal) que cumplen las condiciones de entrada. |
| `prior_needed_human` | entero | De esas, cuántas necesitaron una persona. |
| **`needed_human`** | **0/1** | **Objetivo.** 1 si en la conversación completa (no solo la ventana) hubo traspaso, quedó transferida o abandonada, o escribió una persona del equipo; 0 si la automatización la resolvió sola. |

**Por qué hay vacíos.** Solo tres columnas tienen vacíos en las 62.788 filas:
`first_ai_reply_s` y `ai_chars_mean` están vacías en las **mismas 321 filas**, donde los 4 mensajes
son del cliente y la IA no alcanzó a responder; `cust_gap_mean_s` está vacía en **705**, cuando el
cliente escribió un solo mensaje y no hay pausa que medir. Como el vacío significa algo, el
preprocesamiento crea los indicadores `no_ai_reply` y `single_customer_msg` en vez de imputar a ciegas.

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

#### Cómo usar la interfaz

| Pestaña | Para qué |
|---|---|
| **Conversación** | Se escriben los primeros 4 mensajes como un chat: quién envía cada uno (cliente o atención automática), la hora, el texto y un adjunto opcional. Arriba van el canal, el día y el historial del cliente. La app calcula sola las variables con la misma definición de `sql/extract.sql` y muestra un medidor con el % de riesgo, el semáforo con la acción sugerida y las razones. Tres botones cargan conversaciones inventadas de ejemplo (verde, roja y dudosa). El formulario numérico queda en "Modo avanzado". |
| **Bandeja** | La cola ordenada de mayor a menor riesgo, con semáforo, % y razón principal. Al seleccionar una fila se ve el detalle. Con "Subir un CSV" (las columnas del dataset) devuelve el ranking descargable. |
| **Ejemplos reales (anonimizados)** | 24 conversaciones de octubre, 8 por nivel, con lo que pasó en realidad. Son ilustrativas: el modelo final ya vio octubre. |
| **Cómo funciona el modelo** | Las métricas de octubre en lenguaje simple, el alcance de los datos, la importancia SHAP, el rendimiento por mes, la anticipación y la regla de reajuste. |

Capturas en `figures/app_*.png`. Para comprobar que el cálculo desde el chat coincide con el
extractor: `python tests/test_chat_features.py`.

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
                          chat_features.py (variables desde los mensajes, misma definición que
                          sql/extract.sql) ·
                          monitoring.py (PSI, métricas por empresa, regla de reentrenamiento) ·
                          retrain.py (retador vs. vigente) · baseline.py (prueba inicial) ·
                          export_production.py (export JSON) · timing.py (anticipación).
app/                      app.py (interfaz Streamlit para usuarios no técnicos).
tests/                    test_chat_features.py (chat → variables coincide con el extractor).
models/                   riesgo_v1.joblib · best_params.json · metrics.json · export de producción.
figures/                  Gráficas exportadas por los notebooks · app_*.png (capturas de la interfaz).
docs/                     documento_tecnico.md (documento del proyecto, 3.1–3.9 y matriz de
                          cumplimiento) · propuesta · diccionario de datos · resumen del equipo.
sql/                      extract.sql (extracción original, solo lectura) · timing.sql
                          (escalaciones y anticipación, solo lectura).
```

## Resultados destacados

Prueba temporal: entrenado con todos los datos hasta el 30 de septiembre de 2026, probado con
octubre (2.465 conversaciones nunca vistas, sin la empresa dominante).

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
| 3.8 Despliegue e interfaz | `app/app.py`, `src/chat_features.py`, `tests/`, `models/`, `model_card.md` |
| 3.9 Reentrenamiento | Parte 3; `src/monitoring.py`, `src/retrain.py` |
