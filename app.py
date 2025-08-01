"""
HackRx Webhook API (Sentence Optimized)

- Exposes endpoint: POST /api/v1/hackrx/run
- Downloads & indexes PDF into Pinecone (if not already)
- Runs queries against Gemini + Pinecone
- Returns answers as single sentences
"""

import os
import re
import uuid
import json
import time
import fitz  # PyMuPDF
import requests
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
import asyncio

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# === Load Environment ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-index").lower().replace("_", "-")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# === Configure Clients ===
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    logging.info(f"Index '{INDEX_NAME}' not found. Creating new one...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )
index = pc.Index(INDEX_NAME)

app = FastAPI()

# === Data Models ===
class RunRequest(BaseModel):
    documents: str
    questions: list[str]

# === Helpers ===
def download_pdf(url: str, save_path: str):
    logging.info("📥 Stage 1: Downloading PDF...")
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    logging.info("✅ PDF downloaded")
    return save_path

def extract_text_from_pdf(pdf_path):
    logging.info("📄 Stage 2: Extracting text from PDF...")
    doc = fitz.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text").strip()
        if page_text:
            text += f"\n[Page {page_num}]\n" + page_text + "\n"
    logging.info(f"✅ Extracted {len(text.split())} words from {pdf_path}")
    return text

def chunk_document(text, min_clause_len=80, max_group_size=3, max_chunk_words=500):
    logging.info("✂️ Stage 3: Chunking document...")
    clauses = re.split(r"(?=\n?\s*(\d+(?:\.\d+)*[a-zA-Z]?)\s+)", text)
    chunks, buffer, group_count, current_clause_id = [], [], 0, None

    for clause in clauses:
        if not clause.strip():
            continue
        match = re.match(r"^(\d+(?:\.\d+)*[a-zA-Z]?)\s+", clause)
        if match:
            if buffer:
                content = " ".join(buffer)
                if len(content.split()) > max_chunk_words:
                    parts = [content.split()[i:i+max_chunk_words] for i in range(0, len(content.split()), max_chunk_words)]
                    for p in parts:
                        chunks.append((current_clause_id or f"clause_{len(chunks)+1}", " ".join(p)))
                else:
                    chunks.append((current_clause_id or f"clause_{len(chunks)+1}", content))
                buffer, group_count = [], 0
            current_clause_id = match.group(1)
        buffer.append(clause.strip())
        group_count += 1
        if len(" ".join(buffer).split()) > min_clause_len or group_count >= max_group_size:
            content = " ".join(buffer)
            if len(content.split()) > max_chunk_words:
                parts = [content.split()[i:i+max_chunk_words] for i in range(0, len(content.split()), max_chunk_words)]
                for p in parts:
                    chunks.append((current_clause_id or f"clause_{len(chunks)+1}", " ".join(p)))
            else:
                chunks.append((current_clause_id or f"clause_{len(chunks)+1}", content))
            buffer, group_count = [], 0
    if buffer:
        content = " ".join(buffer)
        if len(content.split()) > max_chunk_words:
            parts = [content.split()[i:i+max_chunk_words] for i in range(0, len(content.split()), max_chunk_words)]
            for p in parts:
                chunks.append((current_clause_id or f"clause_{len(chunks)+1}", " ".join(p)))
        else:
            chunks.append((current_clause_id or f"clause_{len(chunks)+1}", content))

    logging.info(f"✅ Created {len(chunks)} chunks")
    return chunks

def upload_chunks(doc_id, chunks, batch_size=200):
    logging.info("🧠 Stage 4: Checking if doc already exists in Pinecone...")
    existing = index.query(
        vector=[0.0]*768,
        filter={"doc_id": {"$eq": doc_id}},
        top_k=1
    )
    if existing["matches"]:
        logging.info("⚡ Document already indexed. Skipping upload.")
        return

    logging.info("🧠 Uploading new chunks to Pinecone...")
    texts = [chunk for _, chunk in chunks]

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            resp = genai.embed_content(
                model="models/embedding-001",
                content=batch_texts,
                task_type="retrieval_document"
            )
            if isinstance(resp, dict) and "embedding" in resp:
                batch_embeddings = resp["embedding"]
            else:
                batch_embeddings = [e["embedding"] for e in resp]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logging.error(f"❌ Embedding batch failed at {i}: {e}")

    vectors = [
        (str(uuid.uuid4()), emb, {"text": chunk, "clause_id": cid, "doc_id": doc_id})
        for (cid, chunk), emb in zip(chunks, embeddings)
    ]

    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors[i:i+batch_size])

    logging.info(f"✅ Uploaded {len(vectors)} vectors (in batches of {batch_size})")

def embed_query(query):
    return genai.embed_content(
        model="models/embedding-001",
        content="Find insurance coverage details: " + query,
        task_type="retrieval_query"
    )["embedding"]

def retrieve_chunks(query, top_k=10):
    logging.info(f"🔍 Retrieving relevant chunks for query: {query}")
    query_vec = embed_query(query)
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    matches = [m["metadata"]["text"] for m in results["matches"]]
    logging.info(f"✅ Retrieved {len(matches)} chunks for query")
    return matches

async def ask_gemini(query, context_chunks):
    logging.info("🤖 Stage 5: Generating answer with Gemini...")
    context = "\n---\n".join(context_chunks)
    prompt = f"""
You are a health insurance policy analysis assistant.
Answer strictly using the clauses below.

Query: {query}

Policy Clauses:
{context}

Instructions:
- ONLY use given clauses.
- Extract exact numbers/limits.
- If missing, say "Cannot determine".
- Output strictly JSON:

{{
  "decision": "Yes / No / Cannot determine",
  "amount": "Coverage limit or Unknown",
  "justification": "Cite exact clauses",
  "clause_ids": ["C1", "C2"],
  "risk_level": "Red / Orange / Yellow / Black",
  "answers": ["Extracted factual statements"]
}}
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        resp = await asyncio.to_thread(model.generate_content, prompt)
        txt = resp.text.strip()
        if txt.startswith("```json"):
            txt = txt[7:]
        if txt.endswith("```"):
            txt = txt[:-3]
        result = json.loads(txt)
        logging.info("✅ Gemini response generated")
        return result
    except Exception as e:
        logging.error(f"❌ Gemini error: {str(e)}")
        return {
            "decision": "Cannot determine",
            "amount": "Unknown",
            "justification": f"Error: {str(e)}",
            "clause_ids": [],
            "risk_level": "Yellow",
            "answers": []
        }

# === Formatting to single sentence ===
def format_answer(ans: dict) -> str:
    decision = ans.get("decision", "Cannot determine")
    amount = ans.get("amount", "Unknown")
    justification = ans.get("justification", "").strip()
    clauses = ", ".join(ans.get("clause_ids", [])) if ans.get("clause_ids") else "N/A"
    risk = ans.get("risk_level", "Unknown")
    facts = "; ".join(ans.get("answers", []))

    return (
        f"{decision}, {justification} The coverage amount is {amount}, "
        f"under clauses {clauses}, with a {risk} risk level. Facts: {facts}"
    )

# === API Endpoint ===
@app.post("/api/v1/hackrx/run")
async def run_submission(req: RunRequest):
    logging.info("📩 POST request received")
    logging.info(f"➡️ Document: {req.documents}")
    logging.info(f"➡️ Questions: {len(req.questions)}")

    start_time = time.time()
    pdf_path = "temp.pdf"
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, req.documents))

    # Pipeline
    download_pdf(req.documents, pdf_path)
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_document(text)
    upload_chunks(doc_id, chunks)

    tasks = []
    for q in req.questions:
        retrieved = retrieve_chunks(q)
        tasks.append(ask_gemini(q, retrieved))

    raw_answers = await asyncio.gather(*tasks)

    # Convert to sentences
    flat_answers = [format_answer(ans) for ans in raw_answers]

    total_time = round(time.time() - start_time, 2)
    logging.info(f"✅ Completed all processing in {total_time} sec")

    return {"answers": flat_answers}
