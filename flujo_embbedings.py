import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# ========= CONFIGURACIÓN =========
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "problemas-comunes-12152025"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializar cliente Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
# Inicializar cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
index = pc.Index(INDEX_NAME)

df = pd.read_csv("nuevos_problemas.csv", encoding="utf-8")

df["chunk_text"] = df.apply(
    lambda row: f"Cultivo: {row['Cultivo']}. "
                f"Problema: {row['Problema común']}. "
                f"Síntomas: {row['Síntomas']}. "
                f"Causa: {row['Causa principal']}. "
                f"Manejo recomendado: {row['Manejo posible con fertilización, bioestimulación y enmiendas']}.",
    axis=1
)

vectors = []
for i, text in enumerate(df["chunk_text"]):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding
    
    vectors.append({
        "id": str(i),
        "values": emb,
        "metadata": {
            "cultivo": df.iloc[i]["Cultivo"],
            "problema": df.iloc[i]["Problema común"],
            "manejo": df.iloc[i]["Manejo posible con fertilización, bioestimulación y enmiendas"],
            "texto": text
        }
    })


# Insertar en Pinecone
index.upsert(vectors=vectors)
print("Embeddings cargados en Pinecone ✅")


