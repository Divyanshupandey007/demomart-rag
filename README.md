# DemoMart Policy Assistant — Advanced Two-Stage RAG API

A grounded Retrieval-Augmented Generation (RAG) backend service built from scratch using **Python**, **FastAPI**, **Google Gemini**, and **ChromaDB**.

This project implements an **Advanced RAG pipeline with Two-Stage Retrieval (Bi-Encoder Retrieval + LLM Reranking)** to answer customer policy questions with high precision and strict hallucination resistance, built purely with primitives, without LangChain or high-level orchestration wrappers.

---

## 🚀 Key Architectural Highlights

- **Built Without Orchestration Frameworks:** Document ingestion, chunking, embedding generation, vector querying, reranking, and generation prompts were hand-crafted to maintain full visibility over the RAG mechanics.
- **Two-Stage Retrieval (Advanced RAG):**
  - **Stage 1 (Vector Retrieval):** Queries ChromaDB using cosine similarity to fetch the top 10 candidate chunks (high recall).
  - **Stage 2 (LLM Reranking):** Uses Gemini with structured JSON schema outputs (`Pydantic`) to inspect `(query, document)` pairs and select the top 3 most relevant chunks (high precision).
- **Asymmetric Vector Embeddings:** Uses Google Gemini embeddings with dedicated task types (`RETRIEVAL_DOCUMENT` for chunks and `RETRIEVAL_QUERY` for user questions).
- **Prompt Injection Defense & Strict Grounding:** Instructs the LLM to treat retrieved documents as untrusted reference data, refusing to answer if factual evidence is not present in the context.
- **Production-Ready FastAPI Backend:** Features Pydantic v2 validation, source deduplication, interactive OpenAPI/Swagger docs, and clear DTO separation.

---

## 🏗️ Architecture Flow

```text
======================= INGESTION PIPELINE =======================
PDF Documents (documents/)
       ↓
pypdf Extraction (page-by-page)
       ↓
Sliding-Window Chunking (500 chars, 50 chars overlap)
       ↓
Gemini Embeddings (task_type="RETRIEVAL_DOCUMENT")
       ↓
ChromaDB Persistent Store (Cosine Distance HNSW Index)


====================== QUERY & RAG PIPELINE ======================
User Question (POST /ask)
       ↓
Gemini Query Embedding (task_type="RETRIEVAL_QUERY")
       ↓
Stage 1: ChromaDB Vector Search (Top 10 Candidates)
       ↓
Stage 2: Gemini Reranker (Top 3 Chunks via Structured JSON Output)
       ↓
Grounded Generation Prompt (Context + Question + Instructions)
       ↓
Gemini Flash Generation (Strict citations: File & Page)
       ↓
FastAPI JSON Response (Answer + Deduplicated Sources)
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **API Framework:** FastAPI & Uvicorn
- **LLM & Embeddings:** Google Gemini (`gemini-embedding-001`, `gemini-3.1-flash-lite` via the `google-genai` SDK)
- **Vector Database:** ChromaDB (local persistent client)
- **Document Processing:** pypdf
- **Validation & Schemas:** Pydantic v2
- **Environment Management:** python-dotenv

---

## 📁 Project Structure

```text
.
├── documents/               # Input policy PDFs (synthetic corpus)
├── data/                    # ChromaDB persistent storage (git-ignored)
├── ingest.py                # Extraction, chunking, and vector indexing
├── rag.py                   # Retrieval, reranking, and Gemini generation
├── main.py                  # FastAPI application and route handlers
├── requirements.txt         # Project dependencies
├── .env.example             # Template for required environment variables
├── .gitignore
└── README.md
```

---

## ⚡ Getting Started

### 1. Prerequisites

- Python 3.10 or higher
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

### 2. Clone & Set Up a Virtual Environment

```bash
git clone https://github.com/<your-username>/demomart-rag-fastapi.git
cd demomart-rag-fastapi

# Create the virtual environment
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory from the template:

```bash
cp .env.example .env
```

Then add your Gemini API key to `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Ingest Documents

Run the ingestion script once to parse the PDFs, generate embeddings, and populate ChromaDB:

```bash
python ingest.py
```

### 6. Run the FastAPI Server

```bash
uvicorn main:app --reload
```

The server will start at <http://127.0.0.1:8000>.

---

## 📖 API Usage

### Interactive API Docs (Swagger UI)

Visit <http://127.0.0.1:8000/docs> in your browser to test the endpoints directly.

### Endpoints

#### 1. Health Check

`GET /health`

**Response:**

```json
{
  "status": "ok"
}
```

#### 2. Ask a Question

`POST /ask`

**Request body:**

```json
{
  "question": "Can I exchange an item after 30 days?"
}
```

**Response:**

```json
{
  "answer": "A standard change-of-mind exchange is normally unavailable after 30 days (exchange-policy.pdf, page 2). However, defective or warranty-related cases may follow a separate process (exchange-policy.pdf, page 2).",
  "sources": [
    {
      "file": "exchange-policy.pdf",
      "page": 2
    },
    {
      "file": "exchange-policy.pdf",
      "page": 1
    }
  ]
}
```