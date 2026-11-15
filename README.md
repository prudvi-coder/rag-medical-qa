# IBD Literature Research Assistant

A Python + Streamlit retrieval-augmented generation (RAG) prototype for exploring PubMed literature on inflammatory bowel disease. It combines keyword and dense retrieval, then produces answers with links to the retrieved research.

> **Research/education demo only; not medical advice or a clinically validated system.** Sources are abstracts rather than complete medical records or full studies. Answers and citations require human verification.

## At a glance

- **Data pipeline:** PubMed Entrez search → abstract ingestion → cleaned, overlapping chunks
- **Retrieval:** BM25 lexical search + Chroma vector search using PubMedBERT embeddings
- **Fusion:** weighted reciprocal rank fusion (0.4 lexical / 0.6 vector, top 6 chunks)
- **Generation:** query rewriting and context-restricted answers via Groq-hosted Llama
- **Interface:** Streamlit chat with retrieved PubMed references

## Architecture

```text
PubMed -> ingest.py -> data/raw/pubmed_abstracts.json
                    -> chunker.py -> data/raw/chunks.json
                    -> vectorstore.py -> vectorstore/chroma_db/

Question -> rewrite -> BM25 search + PubMedBERT vector search
                   -> reciprocal-rank fusion -> top 6 chunks
                   -> constrained LLM answer + retrieved references
                   -> Streamlit UI (app.py)
```

## Setup

Requirements: Python 3.10+ recommended, internet for PubMed/model downloads, and a Groq API key for LLM requests.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in **your own** credentials. Do not commit `.env`.

```dotenv
GROQ_API_KEY=replace_me
NCBI_EMAIL=your_contact_email@example.com
NCBI_API_KEY=
```

For a fresh index, run in order:

```bash
python ingest.py
python chunker.py
python vectorstore.py
streamlit run app.py
```

The repository currently tracks previously generated JSON abstracts and chunks in `data/raw/`. To rebuild from those snapshots, start at `python vectorstore.py`. The generated Chroma vector index is excluded by `.gitignore`. No deployment or end-to-end accuracy benchmark is claimed.

## Project files

| File | Purpose |
| --- | --- |
| `ingest.py` | PubMed search, abstract retrieval, deduplication and metadata |
| `chunker.py` | Clean and split abstracts with source metadata |
| `vectorstore.py` | Create local Chroma vector index using PubMedBERT embeddings |
| `rag_pipeline.py` | Query rewrite, hybrid retrieval, fusion, LLM response and source references |
| `app.py` | Streamlit UI and retrieved-reference display |
| `check_setup.py` | Offline checks for required data paths and environment variables |

## Limitations and responsible use

- Displayed links correspond to **retrieved references**, not independently fact-checked claims in the generated answer.
- The app's source-coverage indicator counts retrieved chunks. It is **not** a calibrated probability of factual correctness, clinical validity, or source relevance.
- Corpus coverage varies by PubMed query and ingestion date; this is **not** comprehensive medical guidance.
- The test setup does not establish clinical safety or statistical retrieval performance.
- Public abstracts may be subject to publisher licensing restrictions; review redistribution rights before sharing a larger corpus.

## Suggested next improvements

A retrieval benchmark with labeled relevance judgments; an answer-citation consistency evaluation; automated tests; dependency reduction after reproducible install testing; configurable embedding backends; and explicit corpus provenance/licensing documentation.
