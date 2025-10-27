import os
import json
import requests
import numpy as np
import faiss
from tqdm import tqdm

# -------------------------
# CONFIGURATION
# -------------------------
DOC_DIR = "docs"
INDEX_FILE = "faiss.index"
METADATA_FILE = "chunks.json"

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
EMBED_API_URL = "https://integrate.api.nvidia.com/v1/embeddings"

# Confirmed live NVIDIA Embedding Models
NIM_MODEL_1 = "nvidia/nv-embedqa-e5-v5"          
NIM_MODEL_2 = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"

# -------------------------
# TEXT PREPROCESSING
# -------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        chunks.append(text[i:end])
        i += chunk_size - overlap
    return chunks


def load_docs(doc_dir=DOC_DIR):
    docs = []
    for fname in os.listdir(doc_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(doc_dir, fname), encoding="utf-8") as f:
            text = f.read().strip()
        for chunk in chunk_text(text):
            docs.append({"doc": fname, "text": chunk})
    return docs


# -------------------------
# EMBEDDING FUNCTIONS
# -------------------------
def get_embedding_from_nim(text, model_name, input_type="passage"):
    """
    Call NVIDIA embedding NIM via OpenAI-compatible endpoint.
    """
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "input": text,
        "encoding_format": "float",
        "truncate": "NONE",
        "input_type": input_type
    }

    res = requests.post(EMBED_API_URL, headers=headers, json=payload)
    if res.status_code != 200:
        print(f"❌ Error {res.status_code} for {model_name}: {res.text}")
        res.raise_for_status()

    embedding = res.json()["data"][0]["embedding"]
    return embedding


def get_dual_embedding(text):
    """
    Generate embeddings from both models and concatenate.
    """
    emb1 = get_embedding_from_nim(text, NIM_MODEL_1, input_type="passage")
    emb2 = get_embedding_from_nim(text, NIM_MODEL_2, input_type="passage")
    combined = np.array(emb1 + emb2, dtype=np.float32)
    return combined


# -------------------------
# MAIN INGEST PIPELINE
# -------------------------
def main():
    if not NVIDIA_API_KEY:
        print("⚠️ NVIDIA_API_KEY not set. Please export it before running.")
        return

    chunks = load_docs()
    print(f"Loaded {len(chunks)} chunks.")

    all_embeddings = []

    for c in tqdm(chunks, desc="Embedding with dual NIMs"):
        emb = get_dual_embedding(c["text"])
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\n✅ Dual-NIM embeddings stored in:", INDEX_FILE)
    print("✅ Metadata stored in:", METADATA_FILE)


if __name__ == "__main__":
    main()
