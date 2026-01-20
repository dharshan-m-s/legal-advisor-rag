import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

models = client.models.list()

for m in models:
    # safest: print model name
    print("MODEL:", getattr(m, "name", m))

    # print all available attributes (debug)
    try:
        print("FIELDS:", list(m.model_fields.keys()))
    except Exception:
        print("FIELDS: not available")

    print("-" * 50)
