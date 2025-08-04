# import os
# import uuid
# import time
# import requests
# import fitz  # PyMuPDF
# from dotenv import load_dotenv
# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# import google.generativeai as genai
# from pinecone import Pinecone, ServerlessSpec

# # =========================
# # Load environment variables
# # =========================
# load_dotenv()

# PORT = int(os.getenv("PORT", 3001))
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOGGLE_API_KEY")  # fallback

# if not PINECONE_API_KEY or not GOOGLE_API_KEY:
#     raise ValueError("❌ Missing PINECONE_API_KEY or GOOGLE_API_KEY in .env")

# # =========================
# # Initialize Clients
# # =========================
# pc = Pinecone(api_key=PINECONE_API_KEY)
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=768,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1"),
#     )
# index = pc.Index(PINECONE_INDEX_NAME)

# genai.configure(api_key=GOOGLE_API_KEY)

# app = FastAPI()

# # =========================
# # Utility Functions
# # =========================

# def download_pdf(pdf_url: str, out_path: str) -> str:
#     """Download PDF from URL and save locally"""
#     try:
#         resp = requests.get(pdf_url, timeout=30)
#         resp.raise_for_status()
#         with open(out_path, "wb") as f:
#             f.write(resp.content)
#         return out_path
#     except Exception as e:
#         raise RuntimeError(f"❌ Failed to download PDF: {e}")

# def extract_text_from_pdf(pdf_path: str) -> str:
#     """Extract plain text from a PDF file"""
#     text = ""
#     with fitz.open(pdf_path) as doc:
#         for page in doc:
#             text += page.get_text()
#     return text

# def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100):
#     """Split text into overlapping chunks"""
#     chunks, start = [], 0
#     while start < len(text):
#         end = start + max_chars
#         chunks.append(text[start:end])
#         start += max_chars - overlap
#     return chunks

# def embed_and_upsert(doc_id: str, chunks: list):
#     """Embed chunks and upsert into Pinecone"""
#     vectors = []
#     for i, chunk in enumerate(chunks):
#         emb = genai.embed_content(model="models/embedding-001", content=chunk)["embedding"]
#         vectors.append({
#             "id": f"{doc_id}-{i}",
#             "values": emb,
#             "metadata": {"text": chunk}
#         })

#     index.upsert(vectors=vectors)

# def search_pinecone(query: str, top_k: int = 5):
#     """Search Pinecone index for most relevant chunks"""
#     query_emb = genai.embed_content(model="models/embedding-001", content=query)["embedding"]
#     res = index.query(vector=query_emb, top_k=top_k, include_metadata=True)

#     matches = res.get("matches", [])
#     return [m["metadata"].get("text", "") for m in matches]

# def ask_gemini(question: str, context: str) -> str:
#     """Ask Gemini with retrieved context and return one-line answer"""
#     model = genai.GenerativeModel("gemini-1.5-pro")
#     prompt = f"""
# You are an insurance policy assistant.
# Use only the provided policy text to answer.

# Policy Text:
# \"\"\"
# {context}
# \"\"\"

# Question:
# {question}

# Rules:
# - Answer in ONE single line only.
# - Always include exact numbers, % limits, days, rupee amounts if present.
# - If coverage/limit is not explicitly mentioned, respond with exactly: "Not covered" or "Not specified".
# - Do not add justification, explanation, or extra sentences.
# """
#     resp = model.generate_content(prompt)
#     return resp.text.strip() if resp and resp.text else "Not specified"


# # =========================
# # API Routes
# # =========================

# @app.post("/api/v1/hackrx/run")
# async def run_webhook(req: Request):
#     start_time = time.time()
#     try:
#         data = await req.json()
#         pdf_url = data.get("documents")
#         questions = data.get("questions", [])

#         if not pdf_url or not questions:
#             return JSONResponse({"error": "Missing documents or questions"}, status_code=400)

#         # Step 1: Download PDF
#         doc_id = str(uuid.uuid4())
#         pdf_path = f"./{doc_id}.pdf"
#         print("📥 Downloading PDF...")
#         download_pdf(pdf_url, pdf_path)
#         print("✅ PDF downloaded")

#         # Step 2: Extract text
#         print("📄 Extracting text...")
#         text = extract_text_from_pdf(pdf_path)

#         # Step 3: Chunk text
#         print("✂️ Chunking text...")
#         chunks = chunk_text(text)
#         print(f"✅ Created {len(chunks)} chunks")

#         # Step 4: Embed + Upsert
#         print("💾 Embedding & upserting into Pinecone...")
#         embed_and_upsert(doc_id, chunks)
#         print(f"✅ Upserted {len(chunks)} vectors into Pinecone")

#         # Step 5: Answer questions
#         answers = []
#         for q in questions:
#             contexts = search_pinecone(q, top_k=5)
#             context_text = "\n".join(contexts)
#             answer = ask_gemini(q, context_text)
#             answers.append(answer)

#         elapsed = round(time.time() - start_time, 2)
#         return {
#             "answers": answers,
#             "time_taken_sec": elapsed,
#             "debug_info": {"total_chunks": len(chunks), "pinecone_vectors": len(chunks)}
#         }

#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)

# @app.get("/")
# async def root():
#     return {"status": "✅ HackRx Webhook Running", "port": PORT}



import os
import uuid
import time
import hashlib
import fitz  # PyMuPDF
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import logging

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer, CrossEncoder

# =========================
# Setup Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hackrx")

# =========================
# Load environment variables
# =========================
load_dotenv()

PORT = int(os.getenv("PORT", 3001))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY or GEMINI_API_KEY in .env")

# =========================
# Global Configuration
# =========================
MAX_WORKERS = 16
BATCH_SIZE = 100
CHUNK_SIZE = 300     # smaller chunks for precision
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 8
RERANK_TOP_K = 5
EMBEDDING_DIMENSION = 768

# =========================
# Initialize Clients
# =========================
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(PINECONE_INDEX_NAME)

embedding_model = SentenceTransformer('all-mpnet-base-v2')
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    tokenizer = None

embedding_lock = Lock()
upsert_lock = Lock()

app = FastAPI()

# =========================
# Request Model
# =========================
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# =========================
# Utility Functions
# =========================
def stable_doc_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

async def download_pdf_async(pdf_url: str, out_path: str) -> str:
    async with aiohttp.ClientSession() as session:
        async with session.get(pdf_url, timeout=aiohttp.ClientTimeout(total=60)) as response:
            response.raise_for_status()
            content = await response.read()
            with open(out_path, "wb") as f:
                f.write(content)
    return out_path

def extract_text_with_structure(pdf_path: str) -> str:
    full_text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                full_text.append(text.strip())
    return "\n\n".join(full_text)

def simple_chunk_text(text: str, max_chars: int = 1200, overlap: int = 300) -> List[Dict]:
    words = text.split()
    chunks, chunk, length, chunk_id = [], [], 0, 0
    for word in words:
        chunk.append(word)
        length += len(word) + 1
        if length >= max_chars:
            chunks.append({
                'text': " ".join(chunk),
                'chunk_id': chunk_id
            })
            chunk_id += 1
            chunk = chunk[-(overlap//5):]  # approx overlap
            length = sum(len(w)+1 for w in chunk)
    if chunk:
        chunks.append({'text': " ".join(chunk), 'chunk_id': chunk_id})
    return chunks

def embed_chunks(chunks: List[Dict], doc_id: str) -> List[Dict]:
    texts = [c['text'] for c in chunks]
    with embedding_lock:
        embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return [{
        "id": f"{doc_id}-{i}",
        "values": emb.tolist(),
        "metadata": {"text": c['text'], "doc_id": doc_id}
    } for i, (c, emb) in enumerate(zip(chunks, embeddings))]

def upsert_vectors(vectors: List[Dict], doc_id: str):
    with upsert_lock:
        for i in range(0, len(vectors), BATCH_SIZE):
            index.upsert(vectors=vectors[i:i+BATCH_SIZE], namespace=doc_id)

def parallel_embed_and_upsert(doc_id: str, chunks: List[Dict]):
    chunk_batches = [chunks[i:i+50] for i in range(0, len(chunks), 50)]
    all_vectors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(embed_chunks, batch, doc_id) for batch in chunk_batches]
        for f in as_completed(futures):
            all_vectors.extend(f.result())
    upsert_vectors(all_vectors, doc_id)

def enhanced_search(query: str, doc_id: str) -> List[Dict]:
    q_emb = embedding_model.encode([query])[0].tolist()
    res = index.query(vector=q_emb, top_k=TOP_K_RETRIEVAL, include_metadata=True, namespace=doc_id)
    matches = res.get("matches", [])
    results = [{"text": m["metadata"]["text"], "score": m["score"]} for m in matches]

    # Rerank with CrossEncoder
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)
    for i, sc in enumerate(scores):
        results[i]["rerank_score"] = float(sc)
    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:RERANK_TOP_K]

def build_context(results: List[Dict]) -> str:
    return "\n\n".join([f"Context {i+1}: {r['text']}" for i, r in enumerate(results)])

def ask_gemini(question: str, context: str) -> str:
    prompt = f"""
You are an insurance policy assistant. Answer based on contexts.

CONTEXTS:
{context}

QUESTION: {question}

Rules:
- If exact numbers/limits appear, include them.
- If not in contexts, say "Not specified in policy".
- If excluded, say "Not covered".
- Be precise, one sentence.
"""
    try:
        resp = gemini_model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Error"

async def ask_gemini_async(q: str, context: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, ask_gemini, q, context)

# =========================
# API Routes
# =========================
@app.post("/api/v1/hackrx/run")
async def run_webhook(body: QueryRequest):
    start = time.time()
    try:
        pdf_url, questions = body.documents, body.questions
        if not pdf_url or not questions:
            return JSONResponse({"error": "Missing documents or questions"}, status_code=400)

        doc_id = stable_doc_id(pdf_url)
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})

        if doc_id not in namespaces or namespaces[doc_id].get("vector_count", 0) == 0:
            logger.info("📥 Indexing new PDF...")
            pdf_path = f"./{doc_id}.pdf"
            await download_pdf_async(pdf_url, pdf_path)
            text = extract_text_with_structure(pdf_path)
            chunks = simple_chunk_text(text, max_chars=CHUNK_SIZE*4, overlap=CHUNK_OVERLAP*4)
            logger.info(f"Created {len(chunks)} chunks")
            await asyncio.get_event_loop().run_in_executor(None, parallel_embed_and_upsert, doc_id, chunks)
            os.remove(pdf_path)
        else:
            logger.info("⚡ Using cached embeddings")

        tasks = []
        for q in questions:
            results = enhanced_search(q, doc_id)
            ctx = build_context(results)
            tasks.append(ask_gemini_async(q, ctx))

        answers = await asyncio.gather(*tasks)
        elapsed = round(time.time() - start, 2)

        return {
            "answers": answers,
            "time_taken_sec": elapsed,
            "doc_id": doc_id,
            "chunks_processed": namespaces.get(doc_id, {}).get("vector_count", 0),
            "questions_processed": len(questions),
            "avg_time_per_question": round(elapsed/len(questions), 2)
        }
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"status": "✅ HackRx Webhook (Optimized)", "embedding_model": "all-mpnet-base-v2", "reranker": "CrossEncoder-MSMarco", "llm_model": "gemini-1.5-flash"}

@app.get("/health")
async def health_check():
    try:
        stats = index.describe_index_stats()
        test_resp = gemini_model.generate_content("ping")
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "gemini_connected": bool(test_resp.text),
            "total_vectors": stats.get("total_vector_count", 0),
        }
    except Exception as e:
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
