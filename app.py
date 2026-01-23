import os
import shutil
import numpy as np
import faiss
import streamlit as st

from src.extractor import extract_text
from src.chunker import chunk_text
from src.embeddings import embed_text
from src.storage import save_store, load_store
from src.rag import retrieve, generate_answer

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Indian Legal Advisor (RAG)", layout="wide")

# Load store once
if "index" not in st.session_state:
    st.session_state.index, st.session_state.chunks = load_store()

if "chat" not in st.session_state:
    st.session_state.chat = []

st.title("🇮🇳 Indian Legal Advisor (RAG)")
st.caption("Local RAG using FAISS + Embeddings + LLM")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
language = st.sidebar.selectbox("Language", ["English", "Hindi", "Tamil"], index=0)

if st.sidebar.button("🧹 Reset Vectorstore"):
    # Delete vectorstore files
    try:
        if os.path.exists("vectorstore/faiss.index"):
            os.remove("vectorstore/faiss.index")
        if os.path.exists("vectorstore/chunks.json"):
            os.remove("vectorstore/chunks.json")
    except:
        pass

    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.chat = []
    st.sidebar.success("Vectorstore reset done.")

st.divider()

# Upload section
st.subheader("📂 Upload Documents")
uploaded_files = st.file_uploader(
    "Upload PDF / DOCX / TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("⚡ Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            all_chunks = []

            for uf in uploaded_files:
                save_path = os.path.join(UPLOAD_DIR, uf.name)

                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())

                text = extract_text(save_path)

                # Skip scanned/empty PDFs
                if not text.strip():
                    continue

                all_chunks.extend(chunk_text(text))

            if not all_chunks:
                st.error("No extractable text found. (Scanned PDF maybe?)")
            else:
                dim = len(embed_text(all_chunks[0]))
                index = faiss.IndexFlatL2(dim)

                vectors = np.array([embed_text(c) for c in all_chunks])
                index.add(vectors)

                st.session_state.index = index
                st.session_state.chunks = all_chunks

                save_store(index, all_chunks)
                st.success(f"Stored {len(all_chunks)} chunks successfully!")

with col2:
    st.info(f"Chunks loaded: {len(st.session_state.chunks)}")

st.divider()

# Chat section
st.subheader("💬 Ask Legal Questions")

query = st.text_input("Enter your question")

if st.button("Ask"):
    if st.session_state.index is None or not st.session_state.chunks:
        st.error("Upload and process documents first.")
    elif not query.strip():
        st.warning("Enter a question.")
    else:
        retrieved = retrieve(query, st.session_state.index, st.session_state.chunks, k=3)
        answer = generate_answer(query, language, retrieved)

        st.session_state.chat.append(("You", query))
        st.session_state.chat.append(("Assistant", answer))

# Display chat history
for role, msg in st.session_state.chat:
    if role == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Assistant:**\n\n{msg}")
