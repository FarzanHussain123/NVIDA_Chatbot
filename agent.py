import os, json, requests, numpy as np, faiss

INDEX_FILE = "faiss.index"
METADATA_FILE = "chunks.json"

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
API_URL = "https://integrate.api.nvidia.com/v1"

# Models
EMBED_MODEL_1 = "nvidia/nv-embedqa-e5-v5"             # asymmetric (needs input_type)
EMBED_MODEL_2 = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
LLM_MODEL     = "nvidia/llama-3.1-nemotron-nano-8b-v1"

# ---------------- EMBEDDING HELPERS ----------------
def get_embedding_from_nim(text, model_name, input_type="query"):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "input": text,
        "encoding_format": "float",
        "input_type": input_type,
        "truncate": "NONE"
    }
    res = requests.post(f"{API_URL}/embeddings", headers=headers, json=payload)
    if res.status_code != 200:
        print(f"❌ Error {res.status_code} for {model_name}: {res.text}")
    res.raise_for_status()
    return res.json()["data"][0]["embedding"]

def get_dual_embedding(text, for_query=False):
    input_type = "query" if for_query else "passage"
    emb1 = get_embedding_from_nim(text, EMBED_MODEL_1, input_type=input_type)
    emb2 = get_embedding_from_nim(text, EMBED_MODEL_2, input_type=input_type)
    combined = np.array(emb1 + emb2, dtype=np.float32)
    return combined

# ---------------- RETRIEVAL ----------------
def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def retrieve(index, emb, k=3):
    faiss.normalize_L2(emb)
    D, I = index.search(emb, k)
    return I[0], D[0]

# ---------------- LLM COMPLETION ----------------
def ask_llm(prompt, contexts):
    context_text = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)])
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers using the provided context."},
            {"role": "user", "content": f"{prompt}\n\nContext:\n{context_text}"}
        ],
        "temperature": 0.0,
        "max_tokens": 300
    }

    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    res = requests.post(f"{API_URL}/chat/completions", headers=headers, json=payload)
    if res.status_code != 200:
        print("Error:", res.status_code, res.text)
        return "⚠️ LLM call failed."
    return res.json()["choices"][0]["message"]["content"]

# ---------------- MAIN AGENT LOOP ----------------
def main():
    if not NVIDIA_API_KEY:
        print("⚠️ NVIDIA_API_KEY not set.")
        return

    index, chunks = load_index()
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.strip().lower() in ("exit", "quit"):
            break

        emb = get_dual_embedding(q, for_query=True)
        emb = np.expand_dims(emb, axis=0)

        ids, scores = retrieve(index, emb, k=3)
        contexts = [chunks[int(i)] for i in ids]

        answer = ask_llm(q, contexts)
        print("\n=== ANSWER ===\n")
        print(answer)
        print("\n=== SOURCES ===")
        for c in contexts:
            print("-", c["doc"])

if __name__ == "__main__":
    main()
