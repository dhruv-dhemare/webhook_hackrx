from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "hackrx-index").lower().replace("_", "-")

print(f"Deleting index: {index_name} ...")
pc.delete_index(index_name)
print("✅ Index deleted")
