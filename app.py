"""
Webhook Endpoint: /api/v1/hackrx/run

✅ Accepts POST request with:
   {
     "documents": "https://example.com/policy.pdf",
     "questions": ["Is cataract covered?", "Room rent limit?"]
   }

✅ Downloads and extracts PDF text
✅ Splits into clause-aware chunks
✅ Embeds into Pinecone
✅ Uses Gemini for Q&A
✅ Returns structured response
"""

import os
import uuid
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# === Load environment variables ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "hackrx"

# === Init services ===
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # Gemini embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# === FastAPI app ===
app = FastAPI()

# ---- Utility: Extract PDF text ----
def extract_text_from_pdf(url: str):
    response = requests.get(url)
    pdf_path = f"/tmp/{uuid.uuid4()}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(response.content)

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# ---- Utility: Chunk text ----
def chunk_text(text, chunk_size=1000, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ---- Utility: Embed & store ----
def embed_and_store(doc_id: str, chunks: list):
    embed_model = "models/embedding-001"
    for i, chunk in enumerate(chunks):
        emb = genai.embed_content(
            model=embed_model,
            content=chunk
        )["embedding"]

        index.upsert([
            {
                "id": f"{doc_id}-{i}",
                "values": emb,
                "metadata": {"text": chunk}
            }
        ])

# ---- Utility: Retrieve relevant chunks ----
def retrieve_context(query: str, top_k=5):
    emb = genai.embed_content(
        model="models/embedding-001",
        content=query
    )["embedding"]

    results = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return [m["metadata"]["text"] for m in results["matches"]]

# ---- Utility: Gemini Q&A ----
def answer_question(query: str, context_chunks: list):
    prompt = f"""
    You are an insurance policy assistant.
    Question: {query}
    Context from policy:
    {chr(10).join(context_chunks)}

    Provide response as JSON with:
    - decision: Yes/No
    - amount: coverage amount if applicable
    - justification: why
    - clauses: exact clause reference
    - risk_level: Low/Medium/High
    - facts: short bullet facts
    """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

# === Webhook Endpoint ===
@app.post("/api/v1/hackrx/run")
async def run_webhook(req: Request):
    body = await req.json()
    pdf_url = body.get("documents")
    questions = body.get("questions", [])

    if not pdf_url or not questions:
        return JSONResponse({"error": "documents and questions are required"}, status_code=400)

    # Process PDF
    text = extract_text_from_pdf(pdf_url)
    chunks = chunk_text(text)
    doc_id = str(uuid.uuid4())
    embed_and_store(doc_id, chunks)

    # Answer questions
    results = {}
    for q in questions:
        context = retrieve_context(q)
        answer = answer_question(q, context)
        results[q] = answer

    return JSONResponse({"results": results})
