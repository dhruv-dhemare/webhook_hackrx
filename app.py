
import os
import uuid
import time
import requests
import fitz  # PyMuPDF
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# =========================
# Load environment variables
# =========================
load_dotenv()

PORT = int(os.getenv("PORT", 3001))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GOGGLE_API_KEY")  # fallback

if not PINECONE_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("❌ Missing PINECONE_API_KEY or GOOGLE_API_KEY in .env")

# =========================
# Initialize Clients
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(PINECONE_INDEX_NAME)

genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI()

# =========================
# Utility Functions
# =========================

def download_pdf(pdf_url: str, out_path: str) -> str:
    """Download PDF from URL and save locally"""
    try:
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        return out_path
    except Exception as e:
        raise RuntimeError(f"❌ Failed to download PDF: {e}")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF file"""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100):
    """Split text into overlapping chunks"""
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks

def embed_and_upsert(doc_id: str, chunks: list):
    """Embed chunks and upsert into Pinecone"""
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = genai.embed_content(model="models/embedding-001", content=chunk)["embedding"]
        vectors.append({
            "id": f"{doc_id}-{i}",
            "values": emb,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors=vectors)

def search_pinecone(query: str, top_k: int = 5):
    """Search Pinecone index for most relevant chunks"""
    query_emb = genai.embed_content(model="models/embedding-001", content=query)["embedding"]
    res = index.query(vector=query_emb, top_k=top_k, include_metadata=True)

    matches = res.get("matches", [])
    return [m["metadata"].get("text", "") for m in matches]

def ask_gemini(question: str, context: str) -> str:
    """Ask Gemini with retrieved context and return one-line answer"""
    model = genai.GenerativeModel("gemini-1.5-pro")
    prompt = f"""
You are an insurance policy assistant.
Use only the provided policy text to answer.

Policy Text:
\"\"\"
{context}
\"\"\"

Question:
{question}

Rules:
- Answer in ONE single line only.
- Always include exact numbers, % limits, days, rupee amounts if present.
- If coverage/limit is not explicitly mentioned, respond with exactly: "Not covered" or "Not specified".
- Do not add justification, explanation, or extra sentences.
"""
    resp = model.generate_content(prompt)
    return resp.text.strip() if resp and resp.text else "Not specified"


# =========================
# API Routes
# =========================

@app.post("/api/v1/hackrx/run")
async def run_webhook(req: Request):
    start_time = time.time()
    try:
        data = await req.json()
        pdf_url = data.get("documents")
        questions = data.get("questions", [])

        if not pdf_url or not questions:
            return JSONResponse({"error": "Missing documents or questions"}, status_code=400)

        # Step 1: Download PDF
        doc_id = str(uuid.uuid4())
        pdf_path = f"./{doc_id}.pdf"
        print("📥 Downloading PDF...")
        download_pdf(pdf_url, pdf_path)
        print("✅ PDF downloaded")

        # Step 2: Extract text
        print("📄 Extracting text...")
        text = extract_text_from_pdf(pdf_path)

        # Step 3: Chunk text
        print("✂️ Chunking text...")
        chunks = chunk_text(text)
        print(f"✅ Created {len(chunks)} chunks")

        # Step 4: Embed + Upsert
        print("💾 Embedding & upserting into Pinecone...")
        embed_and_upsert(doc_id, chunks)
        print(f"✅ Upserted {len(chunks)} vectors into Pinecone")

        # Step 5: Answer questions
        answers = []
        for q in questions:
            contexts = search_pinecone(q, top_k=5)
            context_text = "\n".join(contexts)
            answer = ask_gemini(q, context_text)
            answers.append(answer)

        elapsed = round(time.time() - start_time, 2)
        return {
            "answers": answers,
            "time_taken_sec": elapsed,
            "debug_info": {"total_chunks": len(chunks), "pinecone_vectors": len(chunks)}
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"status": "✅ HackRx Webhook Running", "port": PORT}
