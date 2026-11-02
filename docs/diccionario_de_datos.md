# Conversaciones en riesgo — diccionario de datos

Una fila = una conversación cuya primera respuesta fue automática (agente de IA o flujo)
en Orvyz (WhatsApp / Instagram), julio 2025 – octubre 2026. 62.788 filas, 12 empresas,
4.347 clientes. El grueso es de julio a octubre de 2026, cuando la atención automática
se generalizó.

Quién respondió primero y si intervino una persona se decide con los mensajes mismos,
no con los contadores de la sesión (el campo "primera respuesta" se registraba mal antes
de septiembre de 2026).

## Anonimización
- Sin texto de mensajes, nombres, teléfonos ni identificadores reales.
- `empresa` y `cliente` son códigos aleatorios; la correspondencia con los ids reales se descartó y no se puede reconstruir.
- Solo empresas con 30 o más conversaciones. Filas en orden aleatorio.
- Las señales de texto se entregan ya convertidas en indicadores 0/1.

## Ventana de observación (evita fuga de información)
Todas las variables salen de los **primeros 4 mensajes** de la conversación, y esos 4 son
anteriores a cualquier mensaje de una persona del equipo. Solo entran conversaciones que
llegan a 4 mensajes sin intervención humana y con al menos 1 mensaje del cliente.

## Columnas
| Columna | Tipo | Descripción |
|---|---|---|
| empresa | categórica | Código de la empresa (E01–E12). Usar para validación por grupos. |
| cliente | categórica | Código del cliente. Un mismo cliente no debe quedar en entrenamiento y prueba. |
| fecha | fecha | Día de inicio (UTC). Para la separación temporal. |
| channel | categórica | whatsapp / instagram. |
| hour_local | entero | Hora de inicio en Colombia (0–23). |
| dow | entero | Día de la semana (1 = lunes … 7 = domingo). |
| w_customer_msgs | entero | Mensajes del cliente en la ventana. |
| w_ai_msgs | entero | Mensajes de la IA (o de un flujo automático) en la ventana. |
| cust_chars_mean | decimal | Largo promedio de los mensajes del cliente (caracteres). |
| cust_chars_max | entero | Largo del mensaje más largo del cliente. |
| cust_question_marks | entero | Signos "?" escritos por el cliente. |
| cust_repeats | entero | Mensajes del cliente idénticos a uno anterior suyo. |
| media_image / media_audio / media_document / media_other | entero | Adjuntos enviados por el cliente, por tipo. |
| first_ai_reply_s | decimal | Segundos entre el primer mensaje del cliente y la primera respuesta automática. Vacío si no hubo. Puede ser negativo si la IA escribió primero. |
| cust_gap_mean_s | decimal | Segundos promedio entre mensajes consecutivos del cliente. Vacío con un solo mensaje. |
| ai_chars_mean | decimal | Largo promedio de las respuestas automáticas. Los menús y botones cuentan su contenido estructurado, por eso hay valores de miles. |
| ai_no_info | 0/1 | La IA dijo que no tenía la información o no podía ayudar. |
| cust_asks_human | 0/1 | El cliente pidió un asesor o una persona. |
| cust_negative_words | 0/1 | El cliente usó palabras de queja. |
| prior_conversations | entero | Conversaciones anteriores del mismo cliente dentro de este conjunto. |
| prior_needed_human | entero | De esas, cuántas necesitaron una persona. |
| **needed_human** | **0/1 (objetivo)** | 1 si en algún momento escribió una persona del equipo, hubo transferencia o el chat quedó transferido/abandonado; 0 si la atención automática lo resolvió sola. |

## Advertencias para el análisis
- **Una empresa domina:** E09 aporta 57.517 filas (92 %) y en ella interviene una persona en el 95 % de los casos, porque su operación es mixta por diseño. Sin E09 quedan 5.271 filas con 55 % de positivos. Conviene reportar resultados con y sin E09 y validar con GroupKFold por empresa.
- **Proporción por empresa muy distinta** (7 % a 95 %): la empresa en sí es un predictor fuerte; un modelo que generalice a empresas nuevas no puede apoyarse en ella.
- **Cobertura temporal:** julio–octubre de 2026 tiene 62.000 filas (4 meses completos); antes hay pocas porque casi no había atención automática. La separación temporal puede hacerse por mes (entrenar jul–sep, probar oct).
- **Mensajes sin enlazar:** entre el 5 % y el 15 % de los mensajes de cada mes no quedaron ligados a su conversación; algunas ventanas pueden tener menos contexto del real.
- **Señales de texto escasas:** sin E09, "pidió asesor" aparece en 30 filas (67 % necesitó persona) y "la IA no tenía la información" en 75 (60 %).
