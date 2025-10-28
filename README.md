# 🧠 NVIDIA Chatbot  

An intelligent **AI chatbot** that uses **FAISS-based retrieval** and **NVIDIA’s Llama 3.1 Nemotron Nano 8B** model for generating human-like, context-aware responses.  

This project demonstrates how to ingest data, create embeddings, and build an AI assistant capable of answering questions based on your own documents.  

---

## ⚙️ Installation & Setup  

Follow these steps to download, configure, and run the chatbot locally:  

```bash
# Step 1️⃣ — Clone the repository from GitHub
git clone https://github.com/<your-username>/NVIDIA_Chatbot.git
cd NVIDIA_Chatbot

# Step 2️⃣ — Create a Python virtual environment (recommended)
python -m venv venv

# Step 3️⃣ — Activate the virtual environment
# ▶ On macOS / Linux:
source venv/bin/activate
# ▶ On Windows:
venv\Scripts\activate

# Step 4️⃣ — Install required dependencies
pip install -r requirements.txt
# Step 5️⃣ — Get and configure your NVIDIA API Key

# 1️⃣  Visit the NVIDIA Build Portal:
#     https://build.nvidia.com/nvidia/llama-3_1-nemotron-nano-8b-v1

# 2️⃣  Sign in with your NVIDIA account (create one if needed).

# 3️⃣  Click “Get API Key” and copy your key.

# 4️⃣  In your project root folder, create a file named `.env`
#     and add the following line inside it:

NVIDIA_API_KEY=your_api_key_here

# Example:
NVIDIA_API_KEY=nvapi_123456789abcdef

# 5️⃣  (Optional) You can also set it manually in your terminal:
# ▶ On Windows:
set NVIDIA_API_KEY=your_api_key_here
# ▶ On macOS / Linux:
export NVIDIA_API_KEY=your_api_key_here
# Step 6️⃣ — Ingest your documents
# This processes files in /docs, chunks text, generates embeddings, and builds a FAISS index.
python ingest.py
# Step 7️⃣ — Launch the chatbot agent
# The chatbot uses NVIDIA’s Llama API with FAISS for retrieval-augmented generation (RAG).
python agent.py
