import os
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

INPUT_FILE = "data/raw/pubmed_abstracts.json"
OUTPUT_FILE = "data/raw/chunks.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 100


def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_abstracts(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_abstracts(abstracts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []

    for abstract in abstracts:
        doc = Document(
            page_content=clean_text(abstract["abstract"]),
            metadata={
                "pmid": abstract["pmid"],
                "title": abstract["title"],
                "journal": abstract["journal"],
                "year": abstract["year"],
                "pubmed_url": abstract["pubmed_url"],
                "chunk_type": "abstract",
            }
        )

        chunks = splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            all_chunks.append(chunk)

    return all_chunks


def save_chunks(chunks, filepath):
    serializable = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def main():
    print(f"Loading abstracts from {INPUT_FILE}...")
    abstracts = load_abstracts(INPUT_FILE)
    print(f"Loaded {len(abstracts)} abstracts")

    print(f"\nChunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}...")
    chunks = chunk_abstracts(abstracts)
    print(f"Total chunks created: {len(chunks)}")

    # Filter out short chunks
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]
    print(f"After filtering short chunks: {len(chunks)}")

    # Chunk quality report
    lengths = [len(c.page_content) for c in chunks]
    avg_len = sum(lengths) // len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    print(f"\n--- Chunk Quality Report ---")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk length: {avg_len} chars")
    print(f"Min chunk length: {min_len} chars")
    print(f"Max chunk length: {max_len} chars")
    print(f"Chunks per abstract (avg): {len(chunks) // len(abstracts)}")

    save_chunks(chunks, OUTPUT_FILE)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()