import os
import json
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

INPUT_FILE = "data/raw/chunks.json"
PERSIST_DIR = "vectorstore/chroma_db"


def load_chunks(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in data
    ]


def build_vectorstore(docs):
    print("Initializing PubMedBERT embeddings (downloading model if first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

    print(f"Building ChromaDB at {PERSIST_DIR}...")
    os.makedirs(PERSIST_DIR, exist_ok=True)

    print(f"Embedding {len(docs)} chunks locally. This may take 2-3 minutes...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="pubmed_crohns"
    )

    return vectorstore


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run chunker.py first.")
        return

    # Clear previous vectorstore
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print("Cleared previous vectorstore.")

    print(f"Loading chunks from {INPUT_FILE}...")
    docs = load_chunks(INPUT_FILE)
    print(f"Loaded {len(docs)} chunks")

    vectorstore = build_vectorstore(docs)

    count = vectorstore._collection.count()
    print(f"\nVectorstore built successfully.")
    print(f"Total vectors stored: {count}")


if __name__ == "__main__":
    main()