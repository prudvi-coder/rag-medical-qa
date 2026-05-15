import streamlit as st
from rag_pipeline import RAGPipeline


def render_result(result):
    """Render confidence badge and citations."""
    level = result["confidence_level"]
    msg = result["confidence_msg"]
    count = result["retrieved_count"]

    if level == "green":
        st.success(f"✅ {msg} ({count} sources)")
    elif level == "yellow":
        st.warning(f"⚠️ {msg} ({count} sources)")
    else:
        st.error(f"❌ {msg} ({count} sources)")

    if result["citations"]:
        with st.expander(f"📚 View {len(result['citations'])} source(s)"):
            for c in result["citations"]:
                st.markdown(
                    f"**{c['title']}**  \n"
                    f"_{c['journal']}_ · {c['year']}  \n"
                    f"[View on PubMed ↗]({c['url']})"
                )
                st.markdown("---")


# Page config
st.set_page_config(
    page_title="IBD Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("🔬 IBD Research Assistant")
    st.markdown("---")
    st.markdown("""
**About this tool:**

Answers questions about Crohn's disease and IBD using 431 peer-reviewed PubMed abstracts.

Every answer includes:
- 📊 Confidence score
- 📚 Source citations  
- 🔗 Direct PubMed links
""")
    st.markdown("---")
    st.markdown("**Example questions:**")
    st.markdown("- What are first-line treatments for Crohn's?")
    st.markdown("- How does Infliximab work?")
    st.markdown("- What is the role of immunosuppressants in IBD?")
    st.markdown("- What are the risks of biologic therapy?")
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize pipeline once
if "pipeline" not in st.session_state:
    with st.spinner("Loading RAG pipeline... (first load takes ~30 seconds)"):
        st.session_state.pipeline = RAGPipeline()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("IBD & Crohn's Disease Research Assistant")
st.caption("Answers grounded in peer-reviewed PubMed literature · Not medical advice")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "result" in message:
            render_result(message["result"])

# Chat input
if prompt := st.chat_input("Ask a question about Crohn's disease or IBD..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching medical literature..."):
            result = st.session_state.pipeline.query(prompt)
        st.markdown(result["answer"])
        render_result(result)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "result": result
    })