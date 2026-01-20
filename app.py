import os
import shutil
import numpy as np
import faiss
import gradio as gr

from src.config import UPLOAD_DIR
from src.extractor import extract_text
from src.chunker import chunk_text
from src.embeddings import embed_text
from src.storage import save_store, load_store
from src.rag import retrieve, generate_answer

os.makedirs(UPLOAD_DIR, exist_ok=True)

STATE = {
    "index": None,
    "chunks": []
}

# Load saved store if available
STATE["index"], STATE["chunks"] = load_store()

def process_files(files):
    all_chunks = []

    for f in files:
        # Save file locally
        file_path = os.path.join(UPLOAD_DIR, os.path.basename(f.name))
        shutil.copy(f.name, file_path)

        text = extract_text(file_path)
        all_chunks.extend(chunk_text(text))

    if not all_chunks:
        return "❌ No text found in uploaded files."

    dim = len(embed_text(all_chunks[0]))
    index = faiss.IndexFlatL2(dim)

    vectors = np.array([embed_text(c) for c in all_chunks])
    index.add(vectors)

    STATE["index"] = index
    STATE["chunks"] = all_chunks

    save_store(index, all_chunks)

    return f"✅ Stored {len(all_chunks)} chunks and saved vectorstore."

def legal_chat(query, language):
    if STATE["index"] is None or not STATE["chunks"]:
        return "❌ Upload and process documents first."

    retrieved = retrieve(query, STATE["index"], STATE["chunks"], k=3)
    return generate_answer(query, language, retrieved)

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("## 🇮🇳 Indian Legal Advisor (Local RAG)")
    gr.Markdown("Gemini 1.5 • FAISS • Multilingual")

    language = gr.Dropdown(["English", "Hindi", "Tamil"], value="English", label="🌐 Language")

    with gr.Tab("📂 Upload & Process"):
        files = gr.File(file_types=[".pdf", ".docx", ".txt"], file_count="multiple")
        btn = gr.Button("Process Documents")
        status = gr.Textbox(label="Status")
        btn.click(process_files, files, status)

    with gr.Tab("💬 Legal Chat"):
        query = gr.Textbox(label="Ask your legal question")
        out = gr.Textbox(lines=10, label="Answer")
        ask = gr.Button("Ask")
        ask.click(legal_chat, [query, language], out)

app.launch()
