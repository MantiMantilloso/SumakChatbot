# Sumak Chatbot

Chatbot agrícola que utiliza embeddings de OpenAI y Pinecone como base de datos vectorial para responder consultas sobre problemas comunes en cultivos (síntomas, causas y manejo recomendado).

## Scripts

| Archivo | Descripción |
|---|---|
| `create_pinecone_index.py` | Crea el índice en Pinecone con las dimensiones necesarias para los embeddings de OpenAI. |
| `flujo_embbedings.py` | Lee los problemas comunes desde CSV, genera embeddings con OpenAI (`text-embedding-3-small`) y los carga en Pinecone. |
| `debugopenai.py` | Script auxiliar para verificar que la API key de OpenAI está configurada correctamente. |

## Configuración

1. Instalar dependencias:
   ```bash
   pip install openai pinecone python-dotenv pandas
   ```

2. Copiar `.env.example` a `.env` y completar con tus claves:
   ```bash
   cp .env.example .env
   ```

3. Ejecutar los scripts en orden:
   ```bash
   python create_pinecone_index.py
   python flujo_embbedings.py
   ```

## Datos

- `problemas_comunes.csv` / `nuevos_problemas.csv` — Catálogo de problemas agrícolas con columnas: Cultivo, Problema común, Síntomas, Causa principal y Manejo recomendado.
