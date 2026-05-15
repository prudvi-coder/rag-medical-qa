import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Verify API key is loaded before doing anything else
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in environment.")
    print("Make sure your .env file exists in this directory and contains:")
    print("  GOOGLE_API_KEY=your_key_here")
    sys.exit(1)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- Embeddings ---
print("Testing Gemini embeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

sample_docs = [
    Document(
        page_content="Crohn's disease is a chronic inflammatory bowel disease affecting the digestive tract.",
        metadata={"source": "test", "pmid": "00001"}
    ),
    Document(
        page_content="Symptoms of Crohn's include abdominal pain, diarrhea, fatigue, and weight loss.",
        metadata={"source": "test", "pmid": "00002"}
    ),
    Document(
        page_content="Treatment options for Crohn's disease include biologics, immunosuppressants, and surgery.",
        metadata={"source": "test", "pmid": "00003"}
    ),
]

# --- ChromaDB ---
print("Building ChromaDB vectorstore...")
vectorstore = Chroma.from_documents(
    documents=sample_docs,
    embedding=embeddings,
    persist_directory="./vectorstore/test"
)

# --- Retrieval ---
print("Testing retrieval...")
query = "What are the symptoms of Crohn's disease?"
results = vectorstore.similarity_search(query, k=2)

print(f"\nQuery: {query}")
print(f"Top results retrieved: {len(results)}")
for i, doc in enumerate(results):
    print(f"\n[{i+1}] {doc.page_content}")
    print(f"    Source PMID: {doc.metadata['pmid']}")

# --- LLM ---
print("\nTesting Groq LLM connection...")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
response = llm.invoke("Say 'RAG pipeline ready' and nothing else.")
print(f"LLM response: {response.content}")

print("\nAll checks passed. Day 1 done.")