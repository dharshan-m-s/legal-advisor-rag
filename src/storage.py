import os
import json
import faiss

from src.config import FAISS_INDEX_PATH, CHUNKS_PATH, VECTOR_DIR

def ensure_dirs():
    os.makedirs(VECTOR_DIR, exist_ok=True)

def save_store(index, chunks):
    ensure_dirs()

    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save chunks
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def load_store():
    """
    Loads FAISS + chunks safely.
    If FAISS index is corrupted, reset store automatically.
    """
    ensure_dirs()

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, []

    # Check file size (common corruption: 0 bytes)
    if os.path.getsize(FAISS_INDEX_PATH) < 100:
        print("⚠️ faiss.index seems corrupted (too small). Resetting store...")
        try:
            os.remove(FAISS_INDEX_PATH)
        except:
            pass
        return None, []

    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        print(f"⚠️ Failed to read FAISS index: {e}")
        print("⚠️ Resetting vectorstore...")
        try:
            os.remove(FAISS_INDEX_PATH)
        except:
            pass
        return None, []

    try:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to read chunks.json: {e}")
        print("⚠️ Resetting chunks...")
        try:
            os.remove(CHUNKS_PATH)
        except:
            pass
        return None, []

    return index, chunks
