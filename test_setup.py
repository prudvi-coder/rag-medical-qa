"""Check the local prerequisites for the current PubMedBERT/Groq RAG pipeline.

No API requests, model downloads, or expensive initialization are performed.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

checks = {
    "Groq key set (required for Q&A)": bool(os.getenv("GROQ_API_KEY")),
    "PubMed chunks available": Path("data/raw/chunks.json").is_file(),
    "Vector index directory exists": Path("vectorstore/chroma_db").is_dir(),
}
for description, passed in checks.items():
    print(f"{'OK' if passed else 'MISSING'}: {description}")

if not all(checks.values()):
    raise SystemExit("One or more prerequisites are missing. See README.md for the setup steps.")

print("Local prerequisites found (not an end-to-end functional test).")
