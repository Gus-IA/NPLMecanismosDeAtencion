# Modelo Seq2Seq con Atención para Traducción Automática (Español ↔ Inglés)

Este repositorio implementa un **modelo de traducción automática basado en redes neuronales recurrentes (GRU)** con un **mecanismo de atención**, siguiendo una arquitectura Encoder–Decoder.  
El proyecto carga un dataset paralelo, prepara vocabularios, construye tensores de entrenamiento, entrena el modelo y finalmente **genera traducciones con visualización de atención**.

---

## 🚀 Características del proyecto

- Limpieza y normalización del texto (remoción de acentos y caracteres no deseados)
- Construcción de vocabularios para ambos idiomas
- Tokenización con `SOS`, `EOS` y `PAD`
- Creación de un `Dataset` y `DataLoader` personalizado
- Arquitectura:
  - **Encoder GRU**
  - **Decoder GRU con atención**
- Entrenamiento paso a paso con:
  - *Teacher forcing implícito mediante uso de predicciones recurrentes*
  - Cálculo de pérdida token por token
- Predicción de traducciones
- Visualización de mapas de atención con `matplotlib`

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
