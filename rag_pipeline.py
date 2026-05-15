import os
import json
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PERSIST_DIR = "vectorstore/chroma_db"
CHUNKS_FILE = "data/raw/chunks.json"

SYSTEM_PROMPT = """You are a medical research assistant specializing in Crohn's disease and inflammatory bowel disease (IBD).

Your job is to answer questions based ONLY on the provided research abstracts.

Rules:
- Answer only from the context provided. Do not use outside knowledge.
- If the context does not contain enough information to answer, say: "I don't have enough information in the retrieved abstracts to answer this confidently."
- Be precise with medical terminology.
- Do not speculate or hallucinate drug names, dosages, or statistics.
- Do NOT invent or guess PMIDs. Citations are handled separately.
- If abstracts contain conflicting clinical approaches, prioritize the most recent publications and note the year.
- You MUST cite the specific context block used for every clinical claim by placing its index number inline (e.g., [1] or [2, 4]).
- Keep answers concise and clinically accurate.

Context from retrieved abstracts:
{context}
"""


class RAGPipeline:
    def __init__(self):
        print("Initializing RAG pipeline...")
        self._load_chunks()
        self._init_retrievers()
        self._init_llm()
        print("RAG pipeline ready.")

    def _load_chunks(self):
        """Load chunks for BM25 retriever."""
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.docs = [
            Document(page_content=item["page_content"], metadata=item["metadata"])
            for item in data
        ]

    def _init_retrievers(self):
        """Initialize hybrid retriever: BM25 + vector."""
        print("  Loading PubMedBERT vectorstore...")
        embeddings = HuggingFaceEmbeddings(
            model_name="NeuML/pubmedbert-base-embeddings"
        )
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="pubmed_crohns"
        )
        self.vector_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = 6
    def _rewrite_query(self, question):
        """Rewrite vague queries into more retrievable medical language."""
        rewrite_prompt = f"""You are a medical search query optimizer for a PubMed literature database.

    Rewrite the following question into a more specific medical search query that will retrieve better results from clinical abstracts. 
    - Use medical terminology
    - Expand abbreviations
    - Add relevant clinical context
    - Keep it concise (under 20 words)
    - Return ONLY the rewritten query, nothing else

    Original question: {question}
    Rewritten query:"""

        response = self.llm.invoke(rewrite_prompt)
        rewritten = response.content.strip()
        return rewritten

    def _hybrid_retrieve(self, question):
        """Manual hybrid retrieval: combine BM25 + vector with RRF."""
        bm25_docs = self.bm25_retriever.invoke(question)
        vector_docs = self.vector_retriever.invoke(question)

        # Reciprocal Rank Fusion
        scores = {}
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 0.4 * (1 / (rank + 60))
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 0.6 * (1 / (rank + 60))

        # Merge all docs, deduplicate, sort by score
        all_docs = {doc.page_content: doc for doc in bm25_docs + vector_docs}
        ranked = sorted(all_docs.keys(), key=lambda k: scores[k], reverse=True)
        return [all_docs[k] for k in ranked[:6]]

    def _init_llm(self):
        """Initialize Groq LLM."""
        print("  Connecting to Groq LLM...")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def _format_context(self, docs):
        """Format retrieved docs into context string with PMID visible to LLM."""
        context_parts = []
        for i, doc in enumerate(docs):
            meta = doc.metadata
            context_parts.append(
                f"[{i+1}] [PMID: {meta.get('pmid', 'Unknown')}] "
                f"{meta.get('title', 'Unknown')} "
                f"({meta.get('journal', 'Unknown')}, {meta.get('year', 'Unknown')})\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(context_parts)

    def _get_confidence(self, docs):
        """
        Confidence scoring based on number of retrieved docs.
        Green:  4+ docs retrieved
        Yellow: 2-3 docs retrieved
        Red:    0-1 docs retrieved
        """
        count = len(docs)
        if count >= 4:
            return "green", "High confidence — multiple relevant sources found."
        elif count >= 2:
            return "yellow", "Medium confidence — limited sources found."
        else:
            return "red", "Low confidence — very few sources found. Answer may be incomplete."

    def _extract_citations(self, docs):
        """Extract unique citations from retrieved docs."""
        seen_pmids = set()
        citations = []
        for doc in docs:
            meta = doc.metadata
            pmid = meta.get("pmid")
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                citations.append({
                    "pmid": pmid,
                    "title": meta.get("title", "Unknown"),
                    "journal": meta.get("journal", "Unknown"),
                    "year": meta.get("year", "Unknown"),
                    "url": meta.get("pubmed_url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                })
        return citations

    def query(self, question):
        """Run a query through the full RAG pipeline."""

        # Retrieve and cap to top 6 combined results
        rewritten_question = self._rewrite_query(question)
        retrieved_docs = self._hybrid_retrieve(rewritten_question)

        # Confidence scoring
        confidence_level, confidence_msg = self._get_confidence(retrieved_docs)

        # Citations
        citations = self._extract_citations(retrieved_docs)

        # Self-correction: if no docs retrieved, skip LLM entirely
        if not retrieved_docs:
            return {
                "rewritten_query": rewritten_question,
                "answer": "I could not find any relevant information in the medical literature database for this query.",
                "citations": [],
                "confidence_level": "red",
                "confidence_msg": "No relevant sources found.",
                "retrieved_count": 0
            }

        # Format context
        context = self._format_context(retrieved_docs)

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        # Run LLM
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": response.content,
            "citations": citations,
            "confidence_level": confidence_level,
            "confidence_msg": confidence_msg,
            "retrieved_count": len(retrieved_docs),
            "rewritten_query": rewritten_question
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()

    test_questions = [
        "What are the first-line treatments for Crohn's disease?",
        "How do biologics work in treating IBD?",
        "What is the role of Infliximab in Crohn's disease?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        result = pipeline.query(question)
        print(f"\nConfidence: [{result['confidence_level'].upper()}] {result['confidence_msg']}")
        print(f"Sources retrieved: {result['retrieved_count']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nCitations ({len(result['citations'])}):")
        for c in result['citations']:
            print(f"  - {c['title']} ({c['year']}) — {c['journal']}")
            print(f"    {c['url']}")