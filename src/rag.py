import numpy as np
from google import genai
from src.config import GOOGLE_API_KEY
from src.embeddings import embed_text

client = genai.Client(api_key=GOOGLE_API_KEY)

def retrieve(query, index, chunks, k=3):
    q_emb = embed_text(query)
    distances, ids = index.search(np.array([q_emb]), k)

    results = []
    for i in ids[0]:
        if 0 <= i < len(chunks):
            results.append((i, chunks[i]))
    return results

def generate_answer(query, language, retrieved):
    if not retrieved:
        return "❌ No relevant legal context found."

    context = "\n\n".join([f"[Chunk {i}]\n{txt}" for i, txt in retrieved])
    sources = ", ".join([f"Chunk {i}" for i, _ in retrieved])

    prompt = f"""
You are an Indian legal advisor.

Rules:
- Respond in {language}
- Use ONLY Indian laws
- Mention Acts and Sections where applicable
- If unsure, say you are not sure
- Be clear and structured

Context:
{context}

User Question:
{query}

At the end, add:
Sources: {sources}
Disclaimer: This is general legal information, not legal advice.
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    return response.text
