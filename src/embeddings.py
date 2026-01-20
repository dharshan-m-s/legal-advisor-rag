import numpy as np
from google import genai
from src.config import GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

def embed_text(text: str) -> np.ndarray:
    emb = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return np.array(emb.embeddings[0].values).astype("float32")
