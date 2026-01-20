import os
import json
import faiss

from src.config import FAISS_INDEX_PATH, CHUNKS_PATH, VECTOR_DIR

def ensure_dirs():
    os.makedirs(VECTOR_DIR, exist_ok=True)

def save_store(index, chunks):
    ensure_dirs()
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def load_store():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, []

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks
