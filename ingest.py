
from pathlib import Path
import os
from pypdf import PdfReader
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "demomart_policies"


def load_document(pdf_dir: Path) -> list[dict]:
    documents = []

    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        reader = PdfReader(pdf_file)

        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            documents.append({
                "file": pdf_file.name,
                "content": text,
                "page": page_no
            })

    return documents


def create_chunks(
    documents: list,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[dict]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []

    for document in documents:
        content = document["content"]
        file = document["file"]
        page = document["page"]

        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + chunk_size
            chunk_text = content[start:end]

            chunks.append({
                "chunk_id": f"{file}_{page}_{chunk_idx}",
                "file": file,
                "content": chunk_text,
                "page": page,
                "chunk_idx": chunk_idx
            })

            start += chunk_size - chunk_overlap
            chunk_idx += 1

    return chunks


def create_embeddings(
    chunks: list,
    client,
    batch_size: int = 32
) -> list[list[float]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    embeddings = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [chunk["content"] for chunk in batch]

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        if (
            not response.embeddings
            or len(response.embeddings) != len(batch)
        ):
            raise RuntimeError("Embeddings were not generated correctly")

        for item in response.embeddings:
            if item.values is None:
                raise RuntimeError("An embedding was empty")

            embeddings.append(item.values)

    return embeddings


def store_in_chroma(chunks: list, embeddings: list) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding")

    chroma_client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    if not chunks:
        raise ValueError("No chunks were generated from the PDFs")

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        metadatas=[
            {
                "file": chunk["file"],
                "page": chunk["page"],
                "chunk_idx": chunk["chunk_idx"]
            }
            for chunk in chunks
        ],
        embeddings=embeddings
    )

    print("Stored records:", collection.count())


if __name__ == "__main__":
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    client = genai.Client(api_key=api_key)

    documents = load_document(BASE_DIR / "documents")
    print("Pages loaded:", len(documents))

    chunks = create_chunks(documents, chunk_size=500, chunk_overlap=50)
    print("Chunks created:", len(chunks))

    embeddings = create_embeddings(chunks, client)
    print("Embeddings generated:", len(embeddings))

    store_in_chroma(chunks, embeddings)
