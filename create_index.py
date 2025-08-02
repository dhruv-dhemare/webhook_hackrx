from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "hackrx-index").lower().replace("_", "-")

print(f"Creating index {index_name} with 768 dimensions ...")

pc.create_index(
    name=index_name,
    dimension=768,   # ✅ Gemini embeddings size
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

print("✅ Index created successfully")
