import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "problemas-comunes-12152025"

pc = Pinecone(api_key=PINECONE_API_KEY)

# List all indexes and check if the desired index exists
existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # OpenAI text-embedding-3-small = 1536
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{INDEX_NAME}' created with 1536 dimensions.")
else:
    print(f"Index '{INDEX_NAME}' already exists.")