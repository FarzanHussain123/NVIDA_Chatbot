import os, json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# === CONFIG ===
INDEX_FILE = "faiss.index"
METADATA_FILE = "chunks.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

# NVIDIA LLM endpoint & key
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "nvidia/llama-3.1-nemotron-nano-8b-v1"

def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def embed_query(text, model):
    emb = model.encode([text], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    return emb

def retrieve(index, emb, k=4):
    D, I = index.search(emb, k)
    return I[0], D[0]

def ask_nvidia_llm(user_prompt, context_docs):
    context_text = "\n\n--- Retrieved Context ---\n"
    for i, c in enumerate(context_docs):
        context_text += f"[{i+1}] {c['text']}\n"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant that answers based on provided documents."},
            {"role": "user", "content": f"{user_prompt}\n\nUse ONLY the context below:\n{context_text}"}
        ],
        "temperature": 0.0,
        "max_tokens": 400
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    res = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
    if res.status_code != 200:
        print("Error:", res.status_code, res.text)
        return "⚠️ API call failed. Check your NVIDIA_API_KEY or model name."
    data = res.json()
    return data["choices"][0]["message"]["content"]

def main():
    model = SentenceTransformer(EMBED_MODEL)
    index, chunks = load_index()
    while True:
        q = input("\nAsk a question (or 'exit'):\n> ")
        if q.strip().lower() in ("exit","quit"): break
        q_emb = embed_query(q, model)
        ids, scores = retrieve(index, q_emb, k=3)
        contexts = [chunks[int(i)] for i in ids]
        if not NVIDIA_API_KEY:
            print("⚠️ NVIDIA_API_KEY not set — cannot query LLM.")
            continue
        answer = ask_nvidia_llm(q, contexts)
        print("\n=== ANSWER ===\n")
        print(answer)
        print("\n=== SOURCES ===")
        for c in contexts:
            print("-", c["doc"])

if __name__ == "__main__":
    main()
