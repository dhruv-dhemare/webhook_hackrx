import os
import sys
import signal
import asyncio
import logging
import structlog
import numpy as np
import uuid
import fitz  # PyMuPDF
import requests
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import redis

# --- Load Env Config ---
load_dotenv()

class Config:
    PORT: int = int(os.getenv("PORT", 3001))
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "hackrx")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "insurance_policies")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    MAX_WORKERS: int = 16
    TOP_K_RETRIEVAL: int = 15
    RERANK_TOP_K: int = 8
    EMBEDDING_DIMENSION: int = 768
    GEMINI_SEMAPHORE_LIMIT: int = 10
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    if not PINECONE_API_KEY or not GEMINI_API_KEY:
        raise ValueError("❌ Missing PINECONE_API_KEY or GEMINI_API_KEY in .env")

# --- Logging ---
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True
)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = structlog.get_logger("hackrx")

# --- External Services ---
executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

# Gemini
genai.configure(api_key=Config.GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(Config.GEMINI_MODEL)

# Pinecone
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
if Config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=Config.PINECONE_INDEX_NAME,
        dimension=Config.EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(Config.PINECONE_INDEX_NAME)

# MongoDB
mongo_client = MongoClient(Config.MONGODB_URI, serverSelectionTimeoutMS=5000)
db = mongo_client[Config.MONGODB_DB_NAME]
documents_collection = db.documents
clauses_collection = db.clauses

# Redis (optional caching)
redis_client = redis.from_url(Config.REDIS_URL, decode_responses=True)

# Models
embedding_model = SentenceTransformer("all-mpnet-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- FastAPI ---
app = FastAPI(title="HackRx Webhook", version="3.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str   # document URL
    questions: List[str]

# --- Helpers ---
def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def multi_strategy_search(query: str, doc_id: str) -> List[Dict]:
    q_emb = embedding_model.encode([query], convert_to_numpy=True)[0].tolist()
    vector_results = index.query(
        vector=q_emb,
        top_k=Config.TOP_K_RETRIEVAL,
        include_metadata=True,
        namespace=doc_id
    )

    clause_ids = [m["id"] for m in vector_results.get("matches", [])]
    if not clause_ids:
        return []

    full_clauses = list(clauses_collection.find({"clause_id": {"$in": clause_ids}}))
    clause_map = {c["clause_id"]: c for c in full_clauses}

    enriched_results = []
    for match in vector_results.get("matches", []):
        cid = match["id"]
        if cid in clause_map:
            c = clause_map[cid]
            enriched_results.append({**c, "vector_score": match["score"]})

    if len(enriched_results) > 1:
        pairs = [(query, r["original_text"]) for r in enriched_results]
        rerank_scores = reranker.predict(pairs)

        vec_scores = np.array([r["vector_score"] for r in enriched_results])
        rer_scores = np.array(rerank_scores)

        # Normalize
        vec_scores = (vec_scores - vec_scores.min()) / (np.ptp(vec_scores) + 1e-8)
        rer_scores = (rer_scores - rer_scores.min()) / (np.ptp(rer_scores) + 1e-8)

        alpha = 0.5 if len(enriched_results) > 5 else 0.4
        for r, vs, rs in zip(enriched_results, vec_scores, rer_scores):
            r["final_score"] = vs * (1 - alpha) + rs * alpha

        enriched_results.sort(key=lambda x: x["final_score"], reverse=True)

    return enriched_results[:Config.RERANK_TOP_K]

def build_context(results: List[Dict], query: str) -> str:
    if not results:
        return "No relevant clauses found."
    context_parts = [f"QUESTION: {query}\n"]
    for i, clause in enumerate(results, 1):
        context_parts.append(
            f"Clause {i} ({clause['section']}): {clause['dense_text']}"
        )
    return "\n".join(context_parts)

def ask_gemini(question: str, context: str) -> str:
    prompt = f"""
You are an expert insurance policy analyst. Answer based only on the provided clauses.
If info is missing, say so explicitly.

CLAUSES:
{context}

QUESTION: {question}
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error("Gemini QA error", error=str(e))
        return f"Error: {str(e)}"

async def ask_gemini_async(question: str, context: str) -> str:
    return await asyncio.get_event_loop().run_in_executor(executor, ask_gemini, question, context)

# --- Unified QA Endpoint ---
@app.post("/api/v1/hackrx/run")
async def qa_webhook(body: QueryRequest):
    # 1. Check if doc exists
    doc = documents_collection.find_one({"url": body.documents})
    if doc:
        doc_id = doc["doc_id"]
    else:
        # --- Ingestion ---
        doc_id = str(uuid.uuid4())
        logger.info("📥 Ingesting document", url=body.documents, doc_id=doc_id)

        # Download PDF
        resp = requests.get(body.documents)
        resp.raise_for_status()
        pdf_bytes = resp.content

        # Extract text
        pdf_doc = fitz.open("pdf", pdf_bytes)
        text = ""
        for page in pdf_doc:
            text += page.get_text("text")

        # Split
        chunks = split_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)

        # Insert document metadata
        documents_collection.insert_one({
            "doc_id": doc_id,
            "url": body.documents,
            "title": body.documents.split("/")[-1],
            "created_at": datetime.utcnow(),
            "processed": True
        })

        # Insert clauses into Mongo + Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            clause_id = str(uuid.uuid4())
            embedding = embedding_model.encode([chunk], convert_to_numpy=True)[0]

            clause_doc = {
                "clause_id": clause_id,
                "doc_id": doc_id,
                "original_text": chunk,
                "dense_text": chunk,
                "section": f"Section-{i+1}",
                "clause_type": "general",
                "keywords": [],
                "created_at": datetime.utcnow()
            }
            clauses_collection.insert_one(clause_doc)

            vectors.append({
                "id": clause_id,
                "values": embedding.tolist(),
                "metadata": {"doc_id": doc_id}
            })

        index.upsert(vectors=vectors, namespace=doc_id)
        logger.info("✅ Ingestion complete", clauses=len(chunks), doc_id=doc_id)

    # 2. Answer questions
    answers = []
    for q in body.questions:
        results = multi_strategy_search(q, doc_id)
        context = build_context(results, q)
        answer = await ask_gemini_async(q, context)
        answers.append(answer)

    return {
        "doc_id": doc_id,
        "answers": answers
    }

# --- Health + Root ---
@app.get("/health")
async def health_check():
    try:
        gemini_test = await asyncio.to_thread(
            gemini_model.generate_content, "What is insurance?"
        )
        return {"status": "healthy", "gemini": bool(gemini_test and gemini_test.text)}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/")
async def root():
    return {"status": "✅ HackRx Webhook v3.2 (Single QA endpoint)"}

# --- Shutdown ---
def handle_shutdown(sig, frame):
    logger.info("🛑 Shutdown", signal=sig)
    try:
        mongo_client.close()
        redis_client.close()
        executor.shutdown(wait=True)
    finally:
        sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting HackRx Webhook v3.2", port=Config.PORT)
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)
