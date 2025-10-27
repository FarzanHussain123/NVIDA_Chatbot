# ingest.py
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from tqdm import tqdm

DOC_DIR = "docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_FILE = "faiss.index"
METADATA_FILE = "chunks.json"

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        i += chunk_size - overlap
    return chunks

def load_docs(doc_dir=DOC_DIR):
    docs = []
    for fname in os.listdir(doc_dir):
        if not fname.endswith(".txt"): continue
        with open(os.path.join(doc_dir, fname), encoding="utf-8") as f:
            text = f.read().strip()
        for chunk in chunk_text(text):
            docs.append({"doc": fname, "text": chunk})
    return docs

def main():
    model = SentenceTransformer(EMBED_MODEL)
    chunks = load_docs()
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL} ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # normalize to use cosine via inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print("Saved index ->", INDEX_FILE)
    print("Saved metadata ->", METADATA_FILE)

if __name__ == "__main__":
    main()
