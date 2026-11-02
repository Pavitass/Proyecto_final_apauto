# Ficha del modelo: riesgo de que una conversación necesite a una persona (`riesgo_v1`)

## Qué hace
Con los **primeros 4 mensajes** de una conversación cuya primera respuesta fue automática (agente de
IA o flujo), y antes de cualquier mensaje de una persona del equipo, estima la probabilidad de que
la conversación termine necesitando a una persona. Devuelve:

- `score`: la probabilidad calibrada (isotónica). Los umbrales se aplican sobre este valor.
- `nivel`: el semáforo. **Rojo / alerta** si `score ≥ 0,467`; **amarillo / vigilar** si `score ≥ 0,295`;
  **verde** si no.
- Las razones en lenguaje simple (SHAP agrupado por concepto), con `RiskModel.shap_values()` y
  `risk_model.explain()`.

## Modelo
- `RandomForestClassifier` (300 árboles, `max_depth=12`, `min_samples_leaf=10`,
  `max_features≈0,61`, `class_weight="balanced"`), imputación con la mediana y **calibración
  isotónica** con 5 pliegues agrupados por cliente.
- Umbrales elegidos con predicciones fuera de pliegue (validación anidada agrupada por cliente):
  **alerta = recall 0,80**, **vigilar = recall 0,95**.
- Entrenado con **5.271 conversaciones de 11 empresas** (julio de 2025 a octubre de 2026, casi todas
  de julio a octubre de 2026). **Excluye a E09**, una empresa con operación mixta por diseño (92 % de
  las filas originales y 95 % de positivos), que dominaría el modelo.
- **No usa `ai_chars_mean`** (largo de las respuestas automáticas). Funciona como huella de la
  empresa: AUC 0,67 mezclando empresas y 0,51 dentro de cada una. Quitarla cuesta 0,02 de PR-AUC en
  octubre y no cambia el orden dentro de cada empresa.
- Artefacto: `models/riesgo_v1.joblib` (clase `RiskModel` de `src/risk_model.py`; para cargarlo,
  `src/` debe estar en el `sys.path`). Hiperparámetros en `models/best_params.json` y métricas
  completas en `models/metrics.json`.

## Métricas
**Prueba temporal (resultado principal):** entrenado con datos hasta septiembre y probado con
octubre de 2026, sin E09, 2.465 conversaciones con 44,5 % de positivos.

| Medida | Octubre | Octubre, solo clientes nuevos |
|---|---|---|
| ROC-AUC | 0,79 | 0,74 |
| PR-AUC (azar = % positivos) | 0,77 (0,44) | 0,78 (0,54) |
| Recall con el umbral de alerta | 0,62 | 0,66 |
| Precisión con el umbral de alerta | 0,71 | 0,70 |
| % de conversaciones alertadas | 39 % | 51 % |

- **Empresas no vistas** (dejando una fuera cada vez): ROC-AUC 0,69. AUC medio dentro de cada
  empresa: 0,64.
- **Calibración en octubre:** riesgo medio predicho 0,46 contra 0,44 real (Brier 0,18). Sin
  corrección por tasa base: el cambio de octubre fue de **mezcla de empresas**, que el modelo ya
  sigue. La corrección EM lo empeora (Brier 0,30).
- **Negocio (octubre):** de 1.096 conversaciones que necesitaron persona, 683 se detectaron al
  cuarto mensaje, antes de cualquier intervención humana. 7 de cada 10 alertas fueron correctas,
  contra 4,4 al azar. En el 10 % más riesgoso, el 96 % necesitó persona.
- **Modelo final** (todos los meses, fuera de pliegue): ROC-AUC 0,81, PR-AUC 0,85, recall 0,80 y
  precisión 0,74 con el umbral de alerta.

## Limitaciones
- **El recall depende de la mezcla de empresas.** La meta de 0,80 se cumplió en validación y no en
  octubre (0,62), porque creció una empresa que escala poco. Para más recall, bajar el umbral
  global, reelegido con el último mes. Los umbrales por empresa probaron ser peores.
- **Dentro de una misma empresa el orden es moderado** (AUC ≈ 0,64, igual con todos los
  algoritmos). El límite es la información de 4 mensajes sin texto, no el modelo.
- **La señal más fuerte es el historial del cliente.** Para clientes nuevos el modelo es más débil.
- Las señales de texto (pidió asesor, queja, la IA no tenía la información) y los adjuntos aparecen
  en menos del 6 % de las conversaciones: aportan poco.
- No mide la anticipación en minutos: falta el momento del primer mensaje humano.
- Canales: solo se entrenó con WhatsApp e Instagram. Otro canal (Messenger, Telegram) entra como
  WhatsApp (`instagram = 0`) y debe vigilarse aparte.
- No se probó en E09 ni en tenants con menos de 30 conversaciones.

## Entradas: cómo calcularlas en Orvyz
**Cuándo puntuar:** cuando la conversación (`conversation_sessions`) llega a su **4.º mensaje** con
estas condiciones:
- el primer mensaje que no es del cliente es automático (`messages.is_from_ai = true`);
- ninguno de los 4 es de una persona del equipo (`NOT is_from_customer AND NOT is_from_ai`);
- al menos 1 es del cliente.

Si una persona escribe antes del 4.º mensaje, la conversación no se puntúa. Las variables usan
**solo esos 4 mensajes**, ordenados por `messages.timestamp`. `build_features()` en
`src/features.py` transforma estos campos crudos en las 32 variables del modelo.

| Campo crudo | Cómo se calcula |
|---|---|
| `channel` | `conversation_sessions.channel`: `whatsapp` / `instagram`. |
| `hour_local` | Hora de `started_at` en `America/Bogota` (0–23). |
| `dow` | Día ISO de `started_at` en `America/Bogota` (1 = lunes … 7 = domingo). |
| `w_customer_msgs` | Mensajes del cliente (`is_from_customer`) entre los 4. |
| `w_ai_msgs` | Mensajes automáticos (`is_from_ai`) entre los 4. El modelo no lo usa (es `4 − w_customer_msgs`), pero se guarda. |
| `cust_chars_mean` / `cust_chars_max` | Promedio y máximo de `length(content)` de los mensajes del cliente (contenido vacío cuenta como 0). |
| `cust_question_marks` | Total de caracteres `?` en los mensajes del cliente. |
| `cust_repeats` | Mensajes del cliente con texto no vacío idéntico a uno anterior suyo dentro de la ventana, tras pasar a minúsculas y colapsar espacios. |
| `media_image` | Mensajes del cliente con `media_type` que empieza por `image`. |
| `media_audio` | `media_type` que empieza por `audio` o `voice`. |
| `media_document` | `media_type` que empieza por `document` o `application`. |
| `media_other` | Cualquier otro `media_type` no nulo distinto de `text`. |
| `first_ai_reply_s` | Segundos entre el primer mensaje del cliente y el primer mensaje automático de la ventana. Vacío si no hubo respuesta automática en los 4. Negativo si el bot escribió primero. |
| `cust_gap_mean_s` | Promedio de segundos entre mensajes consecutivos del cliente en la ventana. Vacío si el cliente escribió uno solo. |
| `ai_no_info` | 1 si algún mensaje automático (minúsculas, espacios colapsados) contiene `no tengo`, `no cuento con`, `no dispongo`, `no puedo ayudarte`, `no tengo acceso` o `no encontr`. |
| `cust_asks_human` | 1 si algún mensaje del cliente contiene `asesor`, `humano`, `una persona`, `agente real` o `hablar con alguien`. |
| `cust_negative_words` | 1 si algún mensaje del cliente contiene `queja`, `reclamo`, `pesimo`, `pésimo`, `terrible`, `molest`, `estafa` o `nadie responde`. |
| `prior_conversations` | Conversaciones **anteriores y ya cerradas** del mismo chat que cumplen las mismas condiciones de entrada. |
| `prior_needed_human` | De esas, cuántas tuvieron `needed_human = 1`. |
| `ai_chars_mean` | Promedio de `length(content)` de los mensajes automáticos. **El modelo no la usa.** Se guarda solo para el monitoreo. |

**Etiqueta (`needed_human`)**, que se calcula al cerrar la conversación y alimenta el
reentrenamiento: 1 si `had_handoff`, si el `status` es `transferred` o `abandoned`, o si hubo algún
mensaje de una persona del equipo; 0 en otro caso.

**Variables derivadas** (las calcula `build_features`): logaritmo con signo de largos y tiempos;
`log1p` de los conteos; indicadores `no_ai_reply` y `single_customer_msg`; `prior_human_rate` y
`has_history`; `has_media`; seno y coseno de la hora; día en one-hot y `weekend`; `instagram`.

## Uso previsto en Orvyz
- **Etiqueta "en riesgo"** en el chat cuando el nivel es rojo, con las razones.
- **Prioridad en la cola:** ordenar por `score` dentro de cada empresa. El orden es útil incluso
  donde la alerta binaria no lo es.
- **Señal para la mejora automática de agentes:** comparar, por agente, la tasa real de
  `needed_human` con el riesgo que se esperaba (`score` promedio). Un agente que escala mucho más de
  lo esperado para sus conversaciones es candidato a revisión. Las razones SHAP agregadas indican
  dónde mirar.
- **Las reglas explícitas van antes que el modelo.** Si el cliente pide un asesor, se transfiere
  de inmediato con el mecanismo que ya existe (`transfer_to_human`). El modelo casi no usa esa señal
  (30 casos en el entrenamiento) y marcarla no cambia el riesgo.
- **No usarlo** para decidir solo, sin una persona, cerrar o desatender conversaciones verdes: el
  recall no es perfecto.

## Monitoreo y reentrenamiento
`src/monitoring.py` y `notebooks/03_monitoreo_reentrenamiento.ipynb`:
- PSI diario de las variables y del puntaje;
- recall, precisión, PR-AUC y calibración mensual por empresa;
- tiempo entre la alerta y el primer mensaje humano.

**Cuándo reentrenar:** si el recall cae más de 5 puntos, si entran empresas nuevas, si hay un
PSI ≥ 0,25, si la descalibración supera 5 puntos, si cambia el modelo de IA de la plataforma, o como
mínimo cada 3 meses.

`src/retrain.py` compara al retador con el vigente en el último mes. El retador reemplaza al
vigente solo si gana en PR-AUC y no pierde más de 2 puntos de recall **alertando el mismo volumen de
conversaciones que el vigente**, y solo después de correr en paralelo. El umbral se revisa en cada
reentrenamiento: con la misma meta de recall, varía unas centésimas según cómo se armen los
pliegues (0,54 o 0,57 con los datos hasta septiembre).

## Modelo de producción (`riesgo_v1_prod`)
`riesgo_v1.joblib` (5 bosques calibrados × 300 árboles más un bosque para explicar, 13 MB) es el
artefacto de investigación. Orvyz evalúa una versión compacta de la misma familia, exportada a JSON
por `src/export_production.py` (`.venv/bin/python -I src/export_production.py`):

- **Un** `RandomForestClassifier` de **60 árboles** con los mismos parámetros
  (`models/best_params.json`), las mismas 32 variables, sin E09 y con imputación por la mediana.
- **Calibración isotónica centrada**, ajustada con predicciones fuera de pliegue agrupadas por
  cliente: cada escalón de la isotónica se reduce a su punto medio y se interpola linealmente entre
  ellos. Una sola isotónica deja ~25 escalones y empata a cientos de conversaciones (costaba 0,02 de
  PR-AUC en octubre); la centrada es estrictamente creciente, conserva el orden del bosque y calibra
  igual.
- **Razones** por el método de caminos de Saabas sobre el mismo bosque, sumadas en los grupos de
  `GROUPS` (no hace falta un bosque aparte).
- Umbrales reelegidos fuera de pliegue (validación anidada agrupada por cliente) con todos los meses:
  **alerta = 0,461** (recall 0,80) y **vigilar = 0,284** (recall 0,95).

| Prueba temporal (entrena ≤ septiembre, prueba octubre, sin E09) | `riesgo_v1` | `riesgo_v1_prod` |
|---|---|---|
| PR-AUC | 0,770 | 0,771 |
| ROC-AUC | 0,786 | 0,788 |
| Brier | 0,184 | 0,184 |
| Riesgo medio predicho (real 0,445) | 0,465 | 0,461 |
| Tamaño | 13 MB (joblib) | 0,68 MB (JSON) |

Fuera de pliegue con todos los meses: PR-AUC 0,847, ROC-AUC 0,811; con el umbral de alerta,
recall 0,80, precisión 0,73 y 60 % de conversaciones alertadas.

Archivos: `models/riesgo_v1_prod.json` (modelo), `models/riesgo_v1_prod_metrics.json` (métricas) y
`models/riesgo_v1_prod_fixtures.json` (60 filas de octubre con casos borde: campos crudos →
variables, puntaje, nivel y peso de cada grupo de razones, calculados con scikit-learn) para la
prueba de paridad del backend. El script verifica además que su evaluador del JSON (el mismo
algoritmo que el backend: entradas en float32 antes de comparar con el umbral, promedio de las hojas
e interpolación) coincide con scikit-learn (diferencia máxima 1,2e-13).
