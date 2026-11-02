# Proyecto A: detección temprana de conversaciones que van a necesitar una persona

## 3.1 Problema
- **Contexto:** Orvyz atiende con IA y flujos automáticos los chats de WhatsApp e Instagram de varios negocios. Algunas conversaciones que empieza la atención automática terminan necesitando a una persona del equipo.
- **Usuarios:** asesores y supervisores de cada negocio, y la propia plataforma, que decide cuándo pasar el chat a una persona.
- **Decisión que apoya:** pasar el chat a una persona antes de que el cliente se frustre, priorizar la cola de atención y detectar qué agentes de IA necesitan ajustes.
- **Datos:** 62.788 conversaciones reales y anonimizadas (`orvyz_conversaciones_riesgo.csv`), de 12 empresas y 4.347 clientes, casi todas de julio a octubre de 2026. El detalle de cada columna está en `diccionario_de_datos.md`.

**Variable objetivo (`needed_human`):** 1 si en algún momento escribió una persona del equipo, hubo transferencia o el chat quedó transferido o abandonado; 0 si la atención automática lo resolvió sola.

**Predicción temprana:** todas las variables salen de los primeros 4 mensajes, y esos 4 son anteriores a cualquier mensaje humano.

**Variables (las mismas para cualquier negocio):**
- mensajes del cliente y de la IA en la ventana, largo promedio y máximo, signos de pregunta, mensajes repetidos;
- tiempo de la primera respuesta automática y tiempo promedio entre mensajes del cliente;
- fotos, audios y documentos enviados por el cliente;
- indicadores de texto: la IA dijo que no tenía la información, el cliente pidió un asesor, el cliente usó palabras de queja;
- historial del cliente: conversaciones anteriores y cuántas necesitaron una persona;
- canal, hora local y día de la semana.

## 3.2 Preprocesamiento
- **Anonimización (ya aplicada):** sin texto, nombres ni teléfonos; empresa y cliente como códigos aleatorios; solo empresas con 30 conversaciones o más.
- **Faltantes:** el tiempo de primera respuesta y el intervalo entre mensajes pueden venir vacíos. Se imputan y se agrega un indicador de "sin dato".
- **Atípicos:** tiempos de horas o días y largos de miles de caracteres (los menús cuentan su contenido estructurado). Se transforman con logaritmo.
- **Codificación:** canal con one-hot, la hora como variable cíclica (seno y coseno) y el día con one-hot.
- **Escalado:** para kNN, SVM y regresión logística; los árboles no lo necesitan.
- **Desbalance:** depende del escenario. Con todas las empresas hay 92 % de positivos; sin la empresa dominante, 55 %. Se usan pesos por clase y se justifica la decisión.

## 3.3 EDA
- Proporción de intervención humana por empresa (va de 7 % a 95 %), canal, hora y día.
- Relación de cada señal con la etiqueta. Sin la empresa dominante, cuando el cliente pide un asesor el 67 % necesita persona, y cuando la IA no tenía la información, el 60 %.
- Distribuciones de los tiempos y largos.
- Peso de la empresa dominante (E09, 92 % de las filas): justifica analizar con y sin ella y validar por empresa.

## 3.4 Paradigma
Aprendizaje supervisado, clasificación binaria. Como extra opcional: K-Medias o DBSCAN para encontrar tipos de conversación.

## 3.5 Modelos y selección
- **Líneas base:** predecir siempre la clase mayoritaria y regresión logística.
- **Comparar:** kNN, SVM, árbol de decisión, Random Forest y Gradient Boosting.
- **Ajuste:** hiperparámetros con validación cruzada por grupos y búsqueda aleatoria.
- **Elección:** por la métrica principal y la estabilidad entre empresas.
- **Explicabilidad:** SHAP global y por predicción.

## 3.6 Generalización
- **Separación por tiempo (resultado principal):** entrenar con julio–septiembre de 2026 y probar con octubre.
- **Separación por empresa:** dejar fuera una empresa a la vez, para medir si sirve con negocios nuevos.
- **Contra la fuga de información:**
  - variables solo de mensajes anteriores a la intervención humana, con ventana fija de 4 mensajes;
  - un mismo cliente nunca queda a la vez en entrenamiento y prueba;
  - preprocesamiento ajustado solo con los datos de entrenamiento.
- **Curvas de aprendizaje,** para mostrar que no hay sobreajuste.

## 3.7 Métricas
- **Recall de la clase "necesita persona":** dejar pasar un caso que sí se complica es lo más costoso.
- **Precisión:** demasiadas alertas cansan al asesor.
- **PR-AUC:** más honesta que la exactitud con clases desbalanceadas.
- **Calibración:** que un "70 % de riesgo" signifique de verdad unos 7 de cada 10.
- **En términos de negocio:** cuántos casos se detectan antes y con cuántos mensajes de anticipación.

### Resultados de una prueba inicial (sin ajustar hiperparámetros)
Script: `baseline.py`.

| Escenario | ROC-AUC | PR-AUC | Azar |
|---|---|---|---|
| Tiempo (jul–sep → oct), sin la empresa dominante | 0,80 | 0,78 | 0,44 |
| Tiempo, todas las empresas | 0,84 | 0,97 | 0,89 |
| Empresas no vistas | 0,65 | 0,69 | 0,55 |

- Hay señal clara sin la empresa dominante.
- Con todas las empresas, la PR-AUC alta se explica por la proporción de positivos (89 %).
- Generalizar a empresas nuevas es lo difícil: el modelo va de 0,82 en unas a casi azar en otras.
- La variable más importante es el largo de las respuestas automáticas, que puede funcionar como huella de cada empresa. Conviene revisarla con SHAP y comparar resultados con y sin ella.

## 3.8 Despliegue e interfaz
- **Interfaz web:** una API pequeña con el modelo y una interfaz en Streamlit o Gradio. Se cargan los datos de una conversación o se elige un ejemplo, y se ve el porcentaje de riesgo, un semáforo y la explicación en lenguaje simple ("el cliente repitió la pregunta 3 veces").
- **En Orvyz:** una etiqueta "en riesgo" en el chat y orden de prioridad en la cola.

## 3.9 Reentrenamiento
- **Etiquetas automáticas:** cada conversación que termina genera su propia etiqueta, así que no hace falta etiquetar a mano.
- **Vigilancia mensual:** recall y PR-AUC por empresa, comparando lo que se predijo con lo que pasó, y cambios en los datos de entrada (PSI).
- **Cuándo reentrenar:** si el recall baja más de 5 puntos, si entran empresas nuevas o si cambia el modelo de IA de la plataforma. Como mínimo, cada 3 meses.
- **Cómo validar el modelo nuevo:** debe ganarle al anterior en las mismas empresas y meses de prueba; primero corre en paralelo sin afectar a nadie y luego reemplaza al anterior, con opción de volver atrás.

**Valor para Orvyz:** alto. Sirve a todas las empresas desde el primer día, mejora con el uso y es lo que hacen Intercom y Zendesk.
