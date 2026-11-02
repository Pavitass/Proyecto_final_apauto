# Proyecto A · Resumen para el equipo

**Repo:** https://github.com/Pavitass/Proyecto_final_apauto

**1. Problema.** Orvyz atiende con IA los chats de WhatsApp e Instagram de varios negocios. Algunas de esas conversaciones terminan necesitando a una persona del equipo. Con solo los **primeros 4 mensajes**, y antes de que intervenga alguien, el modelo predice cuáles van a necesitarla. Así el equipo de atención puede priorizarlas en un semáforo rojo / amarillo / verde.

**2. Datos.** Extracción de solo lectura de la base de Orvyz: 62.788 conversaciones, 12 empresas y 4.347 clientes. Están **anonimizadas**: no hay texto, nombres ni teléfonos, y empresas y clientes son códigos aleatorios.

**Alcance del modelo: empresas y periodo específicos.** El modelo no cubre todos los clientes de Orvyz. Se construyó con una selección acotada:
- **Empresas:** 11 empresas clientes de Orvyz (E01–E08 y E10–E12). La E09, que es el 92 % de los datos y tiene operación mixta, quedó fuera.
- **Clientes finales:** 3.507 personas que escribieron a esas 11 empresas.
- **Periodo:** del 20 de julio de 2025 al 29 de octubre de 2026. El 88 % está entre julio y octubre de 2026: septiembre tiene 1.861 conversaciones y octubre 2.465.
- **Conversaciones:** 5.271, solo las que llegaron a 4 mensajes atendidas por la IA, sin que interviniera una persona.
- **Partición en el tiempo:** se entrena con todo hasta el 30 de septiembre de 2026 (2.806 conversaciones) y se prueba con octubre de 2026 (2.465), un mes que el modelo no vio.

Por eso los resultados valen para estas empresas y este periodo. Si se usara con otras empresas o en otro momento, hay que volver a validarlo; para eso está la estrategia de reajuste.

**3. Modelo.** **Random Forest calibrado** (isotónica). Comparamos dos líneas base y cinco modelos (kNN, SVM, árbol, Random Forest y Gradient Boosting) en cuatro formas de evaluar. El Random Forest ganó o empató en todas, y calibrado su porcentaje es creíble para un usuario no técnico.

**4. Resultados.** Entrenamos con los datos hasta septiembre de 2026 y probamos con octubre de 2026, un mes que el modelo no vio.
- **Octubre:** ROC-AUC **0,79** y PR-AUC **0,77** (el azar es 0,44). De las conversaciones que necesitaron persona, detecta el **62 %**, y el **71 %** de sus alertas son correctas.
- **Clientes nuevos:** ROC-AUC 0,74. **Empresas nunca vistas:** 0,69.
- **Fuga, verificada en producción:** solo 1 de 4.645 conversaciones tenía mensajes posteriores a una escalación dentro de la ventana, y ninguna era de octubre. Las métricas no cambian. Además, las etiquetas barajadas dan 0,48 (azar) y ninguna variable sola pasa de 0,65.
- **Anticipación:** todas las alertas llegan antes de que escriba una persona. En los casos rojos, la mediana es de **27 minutos** de margen, y el 80 % tiene 5 minutos o más.

**5. Despliegue y reajuste.**
- **Despliegue:** app Streamlit para no técnicos, con semáforo y razones en lenguaje simple; hoy corre en local.
- **Reajuste:** monitoreo mensual de cambios en los datos y del recall por empresa. Reentrena si el recall cae, llegan empresas nuevas o pasan 3 meses. El modelo nuevo reemplaza al actual solo si lo supera en la misma prueba.

**6. Dónde está cada cosa.**
- `model_development.ipynb`: el notebook entregable, completo.
- `docs/documento_tecnico.md`: el documento técnico (3.1–3.9, la matriz de cumplimiento y el guion del video).
- `app/app.py`: la interfaz.
- `README.md`: cómo instalar y correr todo.
- Para correr la app:
  ```
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  .venv/bin/streamlit run app/app.py
  ```

**7. Lo que falta** (responsable por definir):
- [ ] Todos: aceptar la invitación al repo.
- [ ] Publicar la app en Streamlit Cloud (share.streamlit.io → conectar el repo → `app/app.py`) y poner el enlace en el README.
- [ ] Grabar el video de 2 minutos. El guion está en el anexo del documento técnico.
- [ ] Preparar la presentación.

**8. Limitaciones para la sustentación.**
- Excluimos E09 (92 % de los datos), así que el modelo representa a las otras 11 empresas.
- El recall de octubre (62 %) no llega a la meta de 80 % que se cumplía en validación. La causa es que cambió la mezcla de empresas; septiembre, con 3 empresas nuevas, fue el peor mes (ROC-AUC 0,54).
- Dentro de una misma empresa el modelo ordena menos (AUC medio de 0,64), y ningún algoritmo lo mejora: el techo lo ponen los datos.
- Solo entran conversaciones que llegan a 4 mensajes sin intervención humana.
- Entre el 5 % y el 15 % de los mensajes no quedaron ligados a su conversación.
- Las señales de texto (pidió asesor, quejas) son muy escasas. Pedir un asesor se atiende con una regla, no con el modelo.
- La anticipación se mide desde el 4.º mensaje, no desde una alerta real. La cola es larga (horas) por los chats que llegan fuera de horario.
- Las razones que muestra la app (SHAP) son correlaciones, no causas.
- La app todavía no está publicada.
