# Documento técnico: detección temprana de conversaciones que van a necesitar una persona

Proyecto final de Machine Learning. Todas las cifras salen de la ejecución de
`model_development.ipynb` (o de los notebooks `notebooks/01–03`, que contiene) sobre
`data/orvyz_conversaciones_riesgo.csv`. Las métricas guardadas están en `models/metrics.json`.

## Resumen

Orvyz atiende con IA y flujos automáticos los chats de WhatsApp e Instagram de varios negocios.
Algunas conversaciones que empieza la atención automática terminan necesitando a una persona del
equipo. Con **los primeros 4 mensajes** de la conversación, todos anteriores a cualquier intervención
humana, un **Random Forest calibrado** estima la probabilidad de que eso ocurra y la traduce en un
semáforo con razones en lenguaje simple.

- **Prueba temporal:** se entrenó con julio–septiembre de 2026 y se probó con octubre, 2.465
  conversaciones que el modelo nunca vio. Da **ROC-AUC 0,79 y PR-AUC 0,77**, contra 0,44 al azar.
  Con clientes nuevos da ROC-AUC 0,74, y con empresas nunca vistas, 0,69.
- **Operación en octubre:** se detectaron **6 de cada 10 casos** al cuarto mensaje y **7 de cada 10
  alertas fueron correctas**, contra 4,4 al azar. En el 10 % más riesgoso, el 96 % necesitó persona.
- **Interfaz:** una aplicación Streamlit para usuarios no técnicos (`app/app.py`).
- **Reajuste:** un ciclo de monitoreo y reentrenamiento probado sobre los meses reales
  (`src/monitoring.py`, `src/retrain.py`).

---

## 3.1 Selección del problema

| | |
|---|---|
| **Contexto** | Orvyz es una plataforma SaaS de atención al cliente para negocios en Colombia (WhatsApp, Instagram). Un agente de IA o un flujo automático responde primero. Cuando no alcanza, interviene una persona del equipo, a menudo tarde, cuando el cliente ya repitió la pregunta o se frustró. |
| **Usuarios** | Asesores y coordinadores de cada negocio, que no son técnicos. También el equipo de Orvyz que mejora los agentes de IA y la propia plataforma, que decide cuándo pasar un chat. |
| **Decisión que apoya** | 1) Pasar el chat a una persona **antes** de que el cliente se frustre (alerta roja). 2) Ordenar la cola de atención por riesgo. 3) Detectar qué agentes automáticos escalan más de lo esperado para sus conversaciones y necesitan ajustes. |
| **Datos** | Extracción de **solo lectura** de la base de producción de Orvyz (`sql/extract.sql`, dentro de `BEGIN READ ONLY`), anonimizada antes de salir: 62.788 conversaciones, 12 empresas y 4.347 clientes, casi todas de julio a octubre de 2026. No incluye texto, nombres ni teléfonos. Empresa y cliente son códigos aleatorios. Diccionario en `docs/diccionario_de_datos.md`. |
| **Etiqueta** | `needed_human = 1` si en algún momento escribió una persona del equipo, hubo transferencia o el chat quedó transferido o abandonado. Si no, 0. Se genera sola al cerrar cada conversación, así que no hace falta etiquetar a mano. |

**Por qué tiene sentido el ML.** Las reglas fijas ("si pide asesor, transferir") cubren muy pocos
casos: pedir un asesor aparece en 30 de 5.271 conversaciones. El riesgo depende de combinaciones de
señales débiles (historial, tiempos, largo y forma de los mensajes, canal, hora), y esas
combinaciones cambian según el negocio. Eso se aprende de datos históricos que ya vienen etiquetados.

**Por qué 4 mensajes.** Es la ventana más corta que deja al menos un intercambio completo
cliente → automatización. Además, solo entran conversaciones que llegan a 4 mensajes sin que escriba
una persona, así que **ninguna variable puede ver la intervención que se quiere predecir** (ver 3.6).

---

## 3.2 Preprocesamiento

Todo el código está en `src/features.py` (transformaciones fila a fila) y en el `Pipeline` de
scikit-learn (lo que se aprende de los datos). Detalle en `model_development.ipynb`, Parte 1, §1 y §8.

| Paso | Qué se hizo | Por qué |
|---|---|---|
| **Anonimización** | Sin texto. Empresa y cliente como códigos aleatorios. Solo empresas con ≥ 30 conversaciones. Las señales de texto llegan ya convertidas en indicadores 0/1. | Privacidad de clientes reales, y la correspondencia con los ids se descartó. Las empresas diminutas no permiten validar por empresa. |
| **Limpieza** | 0 filas duplicadas. Tipos verificados. Filtro de la ventana (4 mensajes, sin humano, ≥ 1 del cliente, primera respuesta automática) aplicado en la extracción. | Garantiza que cada fila sea una predicción hecha en el momento real de uso. |
| **Empresa dominante** | E09 (92 % de las filas, 95 % de positivos, operación mixta por diseño) **se excluye del conjunto principal** y se reporta aparte. Quedan 5.271 filas y 11 empresas. | Con E09, el modelo aprendería "es E09 → persona" y la PR-AUC quedaría inflada por la proporción de positivos (0,97 con 89 % de azar). |
| **Faltantes** | `first_ai_reply_s` y `ai_chars_mean` faltan en las mismas 321 filas (la IA no escribió en la ventana) y `cust_gap_mean_s` en 705 (un solo mensaje del cliente). Se crean los indicadores `no_ai_reply` y `single_customer_msg`, y el valor numérico se imputa con la **mediana ajustada solo con entrenamiento**. `prior_human_rate` sin historial también se imputa. | Los vacíos significan algo (se verificó 1 a 1), así que borrar filas o imputar sin indicador perdería esa señal. La mediana resiste mejor los atípicos que la media. |
| **Atípicos** | Se **conservan**. Los tiempos (de segundos a días) y los largos (miles de caracteres en los menús) pasan por `sign(x)·log(1+|x|)`, y los conteos por `log(1+x)`. | Son reales (clientes que vuelven días después, menús con contenido estructurado), no errores. El logaritmo con signo maneja `first_ai_reply_s` negativos (la IA escribió primero). |
| **Codificación** | Canal → `instagram` 0/1. Hora → seno y coseno. Día → one-hot más `weekend`. Historial → `prior_human_rate` y `has_history`. Adjuntos → `has_media`. | La hora es cíclica (las 23 h quedan junto a las 0 h). El día no es ordinal. La tasa de historial separa "cuánto historial" de "cómo le fue". |
| **Redundancia** | Se descarta `w_ai_msgs` (= 4 − `w_customer_msgs`). | Colineal perfecta. |
| **Variable excluida** | `ai_chars_mean` (largo de las respuestas automáticas) **no entra al modelo final**. | Es una huella de la empresa, no del comportamiento (ver 3.3 y 3.5). |
| **Escalado** | `StandardScaler` dentro del `Pipeline` para regresión logística, kNN y SVM. Los árboles no lo necesitan. | Los modelos de distancia y de margen dependen de la escala. Va dentro del `Pipeline` para ajustarse en cada pliegue solo con su parte de entrenamiento. |
| **Desbalance** | 55/45 sin E09. No se sobremuestrea. Se usa `class_weight="balanced"` y luego se recalibra la probabilidad. | Un desbalance leve no justifica datos sintéticos. El umbral de alerta se elige por criterio de negocio (3.7), no en 0,5. |

**Resultado:** 32 variables numéricas (`build_features(df, drop=("log_ai_chars_mean",))`).

---

## 3.3 Análisis exploratorio (EDA)

Parte 1 de `model_development.ipynb`, figuras `figures/01–07`. Cada hallazgo cambió una decisión:

| # | Hallazgo (con cifra real) | Decisión que tomó |
|---|---|---|
| 1 | **E09 es el 92 % de las filas** y necesita persona en el 95 % de los casos. La proporción por empresa va de 7 % a 95 % (fig. 01). | Conjunto principal sin E09. Métricas por empresa. Validación dejando una empresa fuera. |
| 2 | **El periodo útil es julio–octubre de 2026** (antes casi no había atención automática) (fig. 02). | La separación temporal jul–sep → oct es la evaluación principal. |
| 3 | **La tasa de positivos cae de 64 % (jul–sep) a 44 % (octubre)** porque cambió qué empresas escribieron: E01, que escala poco, pasó a ser el 40 % de octubre. | Hipótesis de cambio de proporción, que se probó y se **refutó** en 3.7: no se corrige la tasa base. Se vigila la calibración por empresa. |
| 4 | **Canal, hora y día son señales débiles.** Instagram escala algo más (fig. 03). | Entran con codificación cíclica / one-hot, sin expectativas. |
| 5 | **Las señales más fuertes vistas de a una** son el historial del cliente, el largo de los mensajes y los tiempos (fig. 04–05). | Transformación logarítmica y variables de historial derivadas. |
| 6 | **`ai_chars_mean` es una huella de la empresa:** ROC-AUC 0,67 mezclando empresas y 0,51 dentro de cada una (fig. 06, 06b). | Se entrena con y sin ella. SHAP lo confirma (67 % de su efecto es la empresa). Se excluye del modelo final. |
| 7 | **El historial sí es señal dentro de cada empresa** (fig. 06b). | Es la base del modelo. Se audita a fondo contra fuga (3.6). |
| 8 | **Las señales de texto y los adjuntos son escasas:** entre 94 % y 100 % de ceros. "Pidió asesor" aparece en 30 filas (67 % positivos, intervalo de Wilson amplio). | Se mantienen por interpretabilidad, pero **pedir asesor se maneja con una regla** en Orvyz, no con el modelo. |
| 9 | **Los faltantes son informativos:** cuando la IA no alcanza a responder, el 86 % necesita persona. | Indicadores de faltante en vez de borrar filas. |
| 10 | **No hay correlaciones extremas** fuera de pares obvios (largo medio y máximo) (fig. 07). | Sin reducción de dimensión. Los árboles toleran la colinealidad moderada. |
| 11 | **36 % de las filas de octubre son de clientes que ya estaban en el entrenamiento.** | Se reporta también la métrica solo con clientes nuevos (1.577 filas). |

---

## 3.4 Paradigma

**Aprendizaje supervisado, clasificación binaria.** La etiqueta existe y se produce sola en cada
conversación cerrada, y la decisión es binaria y graduable (pasar o no a una persona, y cuándo). La
explicabilidad con SHAP es parte del producto: el asesor ve *por qué* un chat está en rojo.

No se combinó con aprendizaje no supervisado. En la propuesta, K-Medias o DBSCAN para tipos de
conversación eran un extra opcional. No se hizo porque el EDA mostró que la estructura dominante de
los datos es la empresa, así que un clustering redescubriría las empresas. Queda como trabajo futuro
si se amplía la ventana con texto.

---

## 3.5 Metodología de entrenamiento y selección del modelo

Parte 2 de `model_development.ipynb`, §1–2, figura `10_comparacion_modelos.png`.

**Protocolo**
1. **Entrenamiento:** julio–septiembre de 2026, 2.806 conversaciones con 65 % de positivos.
   **Prueba:** octubre, 2.465 conversaciones con 44,5 % de positivos. La prueba no se usa para nada
   hasta el final.
2. **Búsqueda aleatoria de hiperparámetros** (`RandomizedSearchCV`, 20 combinaciones por modelo)
   con `StratifiedGroupKFold` de 5 pliegues **agrupado por cliente**, optimizando **PR-AUC**.
3. Cada modelo ganador se evalúa en octubre, en octubre solo con clientes nuevos y dejando una
   empresa fuera a la vez (`LeaveOneGroupOut`). También se mide el AUC medio **dentro** de cada empresa.
4. Todo se repite **con y sin `ai_chars_mean`**.

**Alternativas comparadas** (variante sin `ai_chars_mean`, la desplegada):

| Modelo | PR-AUC CV (± desv.) | ROC-AUC oct | PR-AUC oct | ROC-AUC oct, clientes nuevos | ROC-AUC empresas no vistas | ROC-AUC medio dentro de empresa |
|---|---|---|---|---|---|---|
| Mayoritaria (línea base) | — | 0,500 | 0,445 | 0,500 | — | — |
| Regresión logística (línea base) | 0,830 ± 0,015 | 0,753 | 0,704 | 0,672 | 0,653 | 0,638 |
| kNN | 0,838 ± 0,017 | 0,684 | 0,624 | 0,612 | 0,504 | 0,613 |
| SVM (RBF) | 0,837 ± 0,012 | 0,736 | 0,688 | 0,663 | 0,623 | 0,638 |
| Árbol de decisión | 0,838 ± 0,016 | 0,758 | 0,712 | 0,708 | 0,526 | 0,614 |
| **Random Forest** | **0,872 ± 0,015** | 0,786 | 0,772 | **0,736** | **0,694** | **0,639** |
| Gradient Boosting (HistGB) | 0,866 ± 0,010 | **0,790** | 0,772 | 0,733 | 0,679 | 0,636 |

Con `ai_chars_mean`, Random Forest llega a PR-AUC 0,796 en octubre, pero dentro de cada empresa no
mejora (0,631) y con el árbol las empresas no vistas caen de 0,687 a 0,526. Es decir, la variable
ayuda a *reconocer empresas* y no a ordenar conversaciones.

**Por qué Random Forest sin `ai_chars_mean`**
- Junto con Gradient Boosting, es el mejor en la prueba temporal y con clientes nuevos. La
  diferencia entre los dos es de centésimas.
- Es **el más estable con empresas no vistas** (0,694 contra 0,679), que es el caso de un negocio
  nuevo en la plataforma.
- kNN es el peor: con variables de naturaleza distinta, la distancia no representa la similitud
  entre conversaciones. El árbol solo **se desploma con empresas nuevas** porque sobreajusta a las
  empresas conocidas. SVM y la logística quedan por debajo en todo.
- **Dentro de una empresa todos empatan (≈ 0,64):** ese techo lo ponen los datos (4 mensajes, sin
  texto), no el algoritmo. Elegir un modelo más complejo no lo rompería.
- Quitar `ai_chars_mean` cuesta 0,02 de PR-AUC en octubre y no cambia nada dentro de cada empresa.
  Un modelo que castiga a una empresa por el estilo de su bot no sirve para negocios nuevos.

**Hiperparámetros elegidos** (`models/best_params.json`): `max_depth=12`,
`min_samples_leaf=10`, `max_features≈0,615`, 300 árboles y `class_weight="balanced"`. Una
profundidad acotada y hojas de al menos 10 conversaciones regularizan el bosque. La búsqueda exploró
profundidades de 4 a sin límite, hojas de 1 a 40 y `max_features` de 0,1 a 0,7.

**Pasos posteriores a la selección**
- **Calibración isotónica** (`CalibratedClassifierCV`, pliegues agrupados por cliente): el bosque
  entrenado con pesos balanceados produce probabilidades sesgadas.
- **Umbrales** elegidos con predicciones **fuera de pliegue anidadas**: alerta = recall 0,80 y
  vigilar = recall 0,95.
- **Modelo final:** el mismo procedimiento, reentrenado con los 4 meses
  (`train_risk_model`, `models/riesgo_v1.joblib`). Umbrales finales: **alerta 0,467, vigilar 0,295**.

---

## 3.6 Generalización

**Estrategia: tres esquemas, cada uno responde una pregunta distinta**

| Esquema | Pregunta | Resultado |
|---|---|---|
| **Temporal** (jul–sep → oct) | ¿Funciona el mes siguiente, como en producción? | ROC-AUC **0,786**, PR-AUC **0,770** (azar 0,445) |
| Temporal, solo clientes nuevos | ¿Depende de recordar clientes? | ROC-AUC 0,736, PR-AUC 0,784 (azar 0,543) |
| Dejar una empresa fuera | ¿Sirve para un negocio nuevo? | ROC-AUC 0,694 |
| Validación cruzada agrupada por cliente | Solo para elegir hiperparámetros | PR-AUC 0,872 ± 0,015 |
| Simulación mensual (ago, sep, oct) | ¿Es estable mes a mes? | ROC-AUC 0,81 / 0,54 / 0,79 (septiembre trajo 3 empresas nuevas) |

**Evidencia de que no hay fuga de información** (Parte 2, §3.b de `model_development.ipynb`):

| Vía de fuga | Prueba | Resultado |
|---|---|---|
| Las variables ven la intervención humana | La ventana corta en el primer mensaje humano y solo entran conversaciones con 4 mensajes previos a él (`sql/extract.sql`). | Por construcción, ninguno de los 4 mensajes es humano. |
| El historial incluye la etiqueta propia | Al pasar de una conversación a la siguiente del mismo cliente, `prior_needed_human` sube exactamente la etiqueta de la anterior. | **100 %** de 58.441 pares. La primera conversación de cada cliente tiene historial 0 en el 100 % de los casos. Ninguna incluye fechas posteriores. |
| El historial del mismo día todavía no tenía desenlace | Se reconstruye el historial **solo con días anteriores** y se reentrena. | Octubre: ROC-AUC 0,786 → 0,786 y PR-AUC 0,771 → 0,771. Con clientes nuevos, 0,736 → 0,740. **No cambia.** |
| Dependencia del historial | Se reentrena sin ninguna variable de historial. | PR-AUC 0,771 → 0,731. En las 1.353 conversaciones de octubre sin historial, ROC-AUC 0,72. Hay señal sin él. |
| Señal espuria por la estructura de los datos | Se reentrena con las etiquetas barajadas (5 veces). | ROC-AUC en octubre **0,48** en promedio (azar). |
| Un "atajo" que delata la etiqueta | ROC-AUC de cada variable sola en octubre. | Máximo **0,65** (`prior_conversations`). |
| Filas repetidas entre entrenamiento y prueba | Se cruzan todas las variables. | **0** de 2.465. |
| Preprocesamiento que ve la prueba | Imputación y escalado dentro del `Pipeline`. `build_features` es fila a fila. | Se ajusta solo con entrenamiento en cada pliegue. |
| Hiperparámetros o umbral elegidos con la prueba | Búsqueda y umbrales con validación interna (fuera de pliegue anidada). | Octubre solo se usa para reportar. |
| Un mismo cliente en validación y en entrenamiento | `StratifiedGroupKFold` y calibración agrupados por cliente. | Sin solapamiento dentro de la validación. |

**Desviación respecto a la propuesta, justificada.** La propuesta decía que un mismo cliente nunca
quedaría en entrenamiento y prueba. En la prueba temporal, el 36 % de octubre son clientes que ya
habían escrito, y eso es **legítimo**: en producción el modelo ve clientes que vuelven, y su pasado
entra solo por las variables de historial (verificadas arriba). Para que el resultado no dependa de
eso se reporta también la cifra solo con clientes nuevos. Dentro de la validación cruzada sí se
agrupa por cliente.

**Evidencia de que no hay sobreajuste dañino**
- **Curvas de aprendizaje** (fig. 11): la PR-AUC de validación **sigue subiendo** con más datos y
  varía poco entre pliegues. La brecha con el entrenamiento es la esperable en un bosque. La logística
  casi no tiene brecha porque está limitada por su forma lineal.
- El modelo elegido gana en los tres esquemas, no solo en el que guió la búsqueda.
- **La brecha entre la validación cruzada (0,87) y octubre (0,77) es cambio de distribución, no
  sobreajuste.** Un clasificador adversario distingue julio–septiembre de octubre con ROC-AUC 0,86,
  sobre todo por la mezcla de empresas. Por eso la cifra que se reporta es la de octubre y no la de
  validación.

**Pendiente, solo verificable en producción.** El extractor corta la ventana en el **primer mensaje
humano**, pero no en el momento de una **transferencia** (`had_handoff`). Si el agente transfirió
dentro de los primeros 4 mensajes y una persona escribió después, los mensajes posteriores a la
transferencia estarían en la ventana. La consulta de solo lectura que lo mide ya está escrita
(`sql/timing.sql`): usa `conversation_sessions.handoff_at` y `escalation_events.escalated_at`, y
cuenta cuántos de los 4 mensajes de la ventana son posteriores a la transferencia. No devuelve
identificadores ni texto: las filas se emparejan con el CSV por sus propias variables. Si aparecen
casos, se excluyen (cortar la ventana en la transferencia los deja con menos de 4 mensajes, así que
salen de la población) y se recalculan las métricas de octubre. La consulta fue autorizada por el
equipo, pero los permisos del entorno la bloquearon dos veces; queda lista para correrla.

---

## 3.7 Métricas de desempeño

| Métrica | Por qué | Qué significa en el uso real | Octubre |
|---|---|---|---|
| **PR-AUC** (principal) | Mide la calidad del orden **en la clase que importa** y no se infla con negativos fáciles. Su azar es la proporción de positivos (0,445), así que se compara contra ese número. | Si se revisa la cola de arriba hacia abajo, qué tan "limpios" de casos reales salen los primeros. | **0,770** |
| **ROC-AUC** | Comparable entre empresas con distinta proporción de positivos. | Probabilidad de que un chat que necesitó persona quede más arriba que uno que no. | 0,786 |
| **Recall con el umbral de alerta** | Dejar pasar un caso que se complica es lo más costoso: cliente frustrado o venta perdida. | De cada 10 conversaciones que necesitaron persona, cuántas se marcaron en rojo al cuarto mensaje. | **0,62** (683 de 1.096) |
| **Precisión con el umbral de alerta** | Demasiadas alertas falsas cansan al asesor y la etiqueta deja de usarse. | De cada 10 alertas rojas, cuántas eran reales. | **0,71** (7,1 de 10, contra 4,4 al azar) |
| **% alertado** | Es la carga de trabajo que genera el modelo. | Cuántos chats de cada 100 llegan en rojo. | 39 % |
| **Calibración (Brier, riesgo medio)** | La interfaz muestra un porcentaje. "70 %" debe significar unos 7 de cada 10. | Que el número sea creíble para un no técnico. | Brier 0,184. Riesgo predicho 46,5 % contra 44,5 % real. |
| **Captura en el top-k** | Así se usa la cola de prioridad. | Si se revisa primero el 10 % más riesgoso, el 96 % necesitó persona (captura 22 % de los casos, lift 2,2). | 10 %: precisión 0,96. 20 %: 0,84. 30 %: 0,75. |

**No se usa la exactitud** porque con clases desbalanceadas por empresa (de 6 % a 100 % de
positivos) premia al modelo trivial.

**Elección del umbral.** No se usa 0,5. El umbral de **alerta** se fija para recall 0,80 y el de
**vigilar** para recall 0,95, ambos con predicciones fuera de pliegue. En validación eso dio
recall 0,80 y precisión 0,78. **En octubre el recall bajó a 0,62** porque creció E01, que escala
poco. Se probaron umbrales por empresa: suben el recall a 0,81, pero bajan la precisión a 0,47 y
alertan el 76 % de las conversaciones, peor que moverse sobre la curva global (con recall 0,80, la
precisión global sería 0,63). **Decisión:** un umbral global, reelegido en cada reentrenamiento.

**Calibración y cambio de tasa base.** En octubre la tasa de positivos cayó de 65 % a 44 %. Se probó
la corrección de probabilidades a priori por EM (Saerens): estimó una tasa de 0,13 y empeoró el
Brier de 0,184 a 0,301. La isotónica sin corregir ya acierta, porque el cambio es de **mezcla de
empresas** (covariate shift), que el modelo sigue, y no de proporción dentro de cada empresa.
**Decisión:** no se corrige la tasa base y se vigila la calibración por empresa.

**Anticipación.** La propuesta prometía medir cuántos minutos antes de la intervención humana llega
la alerta. Los datos anonimizados no traen la hora del desenlace, así que la cifra en minutos no se
pudo calcular. Lo que sí está garantizado y medido:

- **En mensajes, la anticipación es total por construcción:** el modelo predice al cerrar el 4.º
  mensaje y la ventana solo admite mensajes anteriores al primer mensaje humano. En el 100 % de las
  conversaciones de octubre la alerta llega **antes** de que escriba una persona. De los 1.096 casos
  que necesitaron persona, el modelo marca a tiempo en rojo 683 (62 %).
- **En minutos, la medición está lista:** `sql/timing.sql` calcula, para cada conversación, los
  segundos entre el 4.º mensaje (momento de la predicción) y la transferencia, la escalación, el
  primer mensaje humano y el cierre. Con eso se reportan la mediana, los percentiles y la fracción de
  alertas que llegan antes del desenlace para cada umbral. No se pudo correr por permisos del
  entorno (ver 3.6). En producción se registra además la hora de la alerta (3.9).

---

## 3.8 Despliegue e interfaz

**Interfaz para usuarios no técnicos: `app/app.py` (Streamlit)**
- **Pestaña "Ejemplos reales (anonimizados)":** 24 conversaciones de octubre, 8 por nivel. Se
  elige una y se ve:
  - el **% de riesgo** con una barra que marca los umbrales;
  - el **semáforo** con la acción sugerida (*Alerta: pasar a una persona ahora*, *Vigilar: subir en
    la cola*, *Bajo: la atención automática probablemente la resuelve*);
  - **"Por qué"**: hasta 5 razones en frases ("De sus 8 conversaciones anteriores, 8 necesitaron a
    una persona", "Sus mensajes son muy cortos"), con flecha ▲/▼ y cuántos puntos mueve cada una;
  - un desplegable **"¿Qué pasó en realidad?"** para contrastar con el desenlace.
- **Pestaña "Ingresar una conversación":** un formulario con controles simples (canal, día, hora,
  cuántos mensajes, largo, fotos y audios, si la IA respondió, si el cliente pidió asesor) que
  recalcula el riesgo al instante.
- **Barra lateral:** qué modelo es, con qué datos se entrenó, su desempeño en octubre en lenguaje
  llano y una advertencia de que las razones son correlaciones, no causas.
- No hace falta escribir código, conocer las variables ni interpretar probabilidades crudas.
- Se verificó con `streamlit.testing.AppTest` (sin excepciones en ambas pestañas) y en el navegador.

**Cómo se ejecuta**
```bash
.venv/bin/streamlit run app/app.py
```

**Artefactos desplegables**
- `models/riesgo_v1.joblib`: un `RiskModel` con el preprocesamiento, el modelo calibrado, los
  umbrales y el explicador.
- `models/riesgo_v1_prod.json`: versión compacta para el backend de Orvyz, 60 árboles y 0,68 MB, con
  el mismo desempeño (PR-AUC 0,771, ROC-AUC 0,788 y Brier 0,184 en octubre). Viene con 60 casos de
  paridad (`riesgo_v1_prod_fixtures.json`).
- `model_card.md`: las entradas que necesita el modelo y **cómo calcular cada una desde los primeros
  4 mensajes**, para integrarlo en la plataforma.

**Uso previsto en Orvyz**
- Etiqueta "en riesgo" (rojo) en el chat, con sus razones.
- Orden de la cola por puntaje dentro de cada empresa.
- Comparación, por agente de IA, de la tasa real de escalamiento contra la esperada por el modelo.
- Las reglas explícitas, como "pidió asesor → transferir", van antes que el modelo.

**Estado:** la interfaz funciona **en local**. Publicarla en una URL (Streamlit Community Cloud o un
servidor) y la integración en la plataforma requieren decisiones del equipo (ver "Pendientes").

---

## 3.9 Estrategia de reajuste

Parte 3 de `model_development.ipynb`, `src/monitoring.py`, `src/retrain.py`. Se **simuló sobre los
meses reales**, como si el modelo hubiera salido a producción el 1 de agosto, el 1 de septiembre y el
1 de octubre.

**1. Cómo detectar degradación o cambio de datos**
- **Sin etiquetas, a diario:** PSI de cada variable y del puntaje del modelo (cuantiles de la
  referencia más un intervalo para vacíos). Por debajo de 0,10 es estable, de 0,10 a 0,25 se vigila y
  desde 0,25 es un cambio fuerte. En octubre el PSI del puntaje fue **0,50** y siete variables
  pasaron de 0,25 (tiempo de primera respuesta 0,65, tasa de historial 0,61, largo 0,39…). En
  septiembre fue **0,72**. La alarma llega antes que la caída del recall.
- **Con etiquetas, al cerrar el mes:** recall, precisión, PR-AUC, % alertado y calibración
  (predicho contra real), **por empresa**, con un mínimo de 30 casos para reportar. En producción se
  suma la anticipación (alerta → primer mensaje humano).

**2. Criterio y frecuencia para reentrenar** (`retrain_decision`). Se revisa cada mes y se reentrena si
se cumple cualquiera de estas condiciones:
- el recall cae más de 5 puntos frente a la referencia;
- entran empresas nuevas;
- alguna variable tiene PSI ≥ 0,25;
- la calibración se desvía más de 5 puntos;
- cambia el modelo de IA de la plataforma;
- pasan 3 meses sin reentrenar.

En la simulación de octubre la regla pidió reentrenar: el recall bajó de 0,80 a 0,58 y siete
variables superaron PSI 0,25.

**Evidencia de que la regla de empresas nuevas es necesaria.** Septiembre fue el peor mes (ROC-AUC
0,54) porque llegaron 3 empresas que el modelo de agosto nunca vio. Con un negocio nuevo, el modelo
se usa solo para ordenar la cola hasta tener un mes de datos suyos.

**3. Cómo validar y poner en producción la nueva versión** (`src/retrain.py`)
1. El **retador** se entrena con todo lo anterior al último mes cerrado, con el mismo
   procedimiento.
2. Se compara con el **vigente** en ese mes, con las mismas conversaciones y empresas.
3. **Regla de promoción:** el retador debe ganar en PR-AUC y no perder más de 2 puntos de recall
   **alertando el mismo volumen de conversaciones que el vigente**, es decir, con la misma carga de
   trabajo para el equipo. La comparación ingenua (cada modelo con su propio umbral) rechazaba a un
   retador mejor, y por eso se corrigió.
4. Si gana, se reentrena con todo, incluido el último mes, y corre **en paralelo** una o dos
   semanas: se guardan ambos puntajes sin mostrar el nuevo.
5. Luego reemplaza al vigente. El artefacto anterior se conserva para **volver atrás** con un
   cambio de configuración. El umbral se reelige en cada versión.

**Resultado en la simulación.** El vigente (1 de septiembre) contra el retador (1 de octubre) en
octubre: PR-AUC **0,657 → 0,770** y recall a igual volumen **0,647 → 0,705**. Se promueve.

**Cuidados**
- No reentrenar con conversaciones abiertas, porque su etiqueta no es definitiva.
- El historial solo cuenta conversaciones cerradas antes de la actual.
- Cada puntaje se guarda con la versión del modelo.

---

## Entregables y matriz de cumplimiento

Estado al 31 de octubre de 2026. Las celdas citan dónde está la evidencia.

| Requisito | Estado | Evidencia | Qué falta |
|---|---|---|---|
| **3.1 Problema** | Cumple | Este documento §3.1; `docs/propuesta_proyecto_A.md`; introducción de `model_development.ipynb`. | — |
| **3.2 Preprocesamiento** | Cumple | §3.2 (tabla paso → por qué); `src/features.py`; notebook Parte 1 §1 y §8. Faltantes verificados uno a uno; imputación y escalado en el `Pipeline`. | — |
| **3.3 EDA profundo** | Cumple | Notebook Parte 1, §1–10, 8 figuras; 11 hallazgos con cifras, cada uno ligado a una decisión (§3.3). Incluye intervalos de Wilson para señales escasas y AUC dentro de empresa. | — |
| **3.4 Paradigma** | Cumple | Supervisado, clasificación binaria; no supervisado descartado con razón. | Opcional: clustering si se agrega texto. |
| **3.5 Selección del modelo** | Cumple | 2 líneas base y 5 candidatos (kNN, SVM, árbol, RF, GB), cada uno con búsqueda aleatoria agrupada, × 2 variantes, en 4 esquemas; elección argumentada; hiperparámetros en `models/best_params.json`. | — |
| **3.6 Generalización** | Cumple, con una verificación pendiente | Prueba temporal, clientes nuevos, empresas no vistas, simulación mensual, curvas de aprendizaje y **auditoría de fuga** (10 pruebas, notebook Parte 2 §3.b). | Correr `sql/timing.sql` para descartar transferencias dentro de la ventana (autorizada, bloqueada por permisos). |
| **3.7 Métricas** | Cumple, con una parcial | PR-AUC, ROC-AUC, recall, precisión, % alertado, calibración y top-k, cada una con su significado; umbral por criterio de negocio; política por empresa evaluada y descartada. | Anticipación en mensajes garantizada (100 % antes del humano); en minutos, consulta lista (`sql/timing.sql`) pero bloqueada por permisos. |
| **3.8 Despliegue e interfaz** | Parcial | App Streamlit funcional y probada para no técnicos; artefactos `joblib` y JSON de producción; `model_card.md` con el contrato de entradas. | **Publicarla en una URL** o integrarla en Orvyz, para que la usen clientes reales. Ambas son decisiones del equipo. |
| **3.9 Reajuste** | Cumple | Detección (PSI y métricas por empresa), criterio y frecuencia (`retrain_decision`), validación retador/vigente a igual volumen y despliegue en paralelo con vuelta atrás; simulado sobre agosto–octubre. | — |
| **Notebook `model_development.ipynb`** | Cumple | En la raíz del repo: une EDA, preprocesamiento, modelado, fuga, métricas, SHAP y reajuste con justificaciones; ejecutado de principio a fin sin errores. | — |
| **Repositorio** | Parcial | Carpeta lista: README con correspondencia al enunciado, `.gitignore` y código en `src/`. | Hoy **no es un repositorio git** ni tiene enlace. Hay que decidir hosting y visibilidad: el CSV anonimizado pesa 5,4 MB y el modelo 13 MB. |
| **Video de 2 minutos** | No cumple | Guion en el anexo. | Grabarlo (lo hace el equipo). |
| **Presentación** | No aplica todavía | Este documento sigue el orden que pide la evaluación. | Preparar las diapositivas, si se piden. |

## Pendientes y decisiones del equipo

1. **Grabar el video** (guion abajo).
2. **Crear el repositorio y su enlace.** Hay que decidir:
   - si va público o privado;
   - si se sube el CSV anonimizado, que no tiene texto ni identificadores reales pero es de clientes
     de Orvyz;
   - si se sube el `.joblib` de 13 MB o se regenera con el notebook.
3. **Publicar la interfaz** para que la usen personas reales. Las opciones son:
   - Streamlit Community Cloud, que exige el repositorio en GitHub;
   - un servidor propio;
   - la integración en Orvyz con `riesgo_v1_prod.json`, que es el despliegue real para clientes y ya
     tiene el contrato en `model_card.md`.
4. **Correr `sql/timing.sql` (solo lectura) mes a mes, de julio a octubre de 2026.** Resuelve dos cosas
   a la vez: la fuga por transferencia dentro de la ventana (3.6) y la anticipación en minutos (3.7).
   Se corre con `psql "$URL_DE_LECTURA" -f` tras reemplazar `@FROM@` y `@TO@`. Si hay transferencias
   antes del 4.º mensaje, se excluyen esas filas, se recalculan las métricas y se reentrena.
   El equipo la autorizó, pero los permisos del entorno la bloquearon.

## Anexo: guion del video (2 minutos)

| Tiempo | Qué se muestra | Qué se dice |
|---|---|---|
| 0:00–0:15 | Un chat de WhatsApp atendido por la IA | "En Orvyz la IA responde primero. Algunas conversaciones terminan necesitando a una persona, y llegar tarde cuesta clientes." |
| 0:15–0:30 | El esquema de los 4 mensajes | "Con solo los primeros 4 mensajes, antes de que intervenga nadie, el modelo estima ese riesgo." |
| 0:30–1:05 | La app, pestaña de ejemplos: un caso rojo | Riesgo, semáforo y razones: "De sus 8 conversaciones anteriores, 8 necesitaron persona". Abrir "¿Qué pasó en realidad?". |
| 1:05–1:20 | Un caso verde | "Mensajes cortos, historial resuelto solo: la IA lo resuelve sola." |
| 1:20–1:35 | Pestaña manual: cambiar el historial y ver moverse el riesgo | "Cualquier asesor puede probarlo sin saber de modelos." |
| 1:35–1:50 | Barra lateral con las métricas | "Probado en octubre, un mes que no vio: 7 de cada 10 alertas son correctas y el 10 % más riesgoso acierta 96 %." |
| 1:50–2:00 | La figura del seguimiento mensual | "Cada mes se vigila el cambio de datos y se reentrena solo si el modelo nuevo gana con la misma carga de alertas." |
