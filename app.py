import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import docx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# -------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------
st.set_page_config(page_title="Knowledge Base Agent", layout="wide")

# -------------------------------------------------------------
# API KEY
# -------------------------------------------------------------
API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    st.error("Please set GEMINI_API_KEY or GOOGLE_API_KEY.")
    st.stop()

genai.configure(api_key=API_KEY)

# -------------------------------------------------------------
# Models
# -------------------------------------------------------------
GEN_MODEL = genai.GenerativeModel("models/gemini-2.5-flash")
EMBED_MODEL = "models/text-embedding-004"

# -------------------------------------------------------------
# Vector Store
# -------------------------------------------------------------
VECTOR_STORE = []
DOC_STORE = []

# -------------------------------------------------------------
# Session State
# -------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def extract_text_from_pdf(path):
    text = []
    try:
        reader = PdfReader(path)
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text.append(t)
    except Exception as e:
        st.error(f"PDF error: {e}")
    return "\n".join(text)


def extract_text_from_docx(path):
    text = []
    try:
        d = docx.Document(path)
        for p in d.paragraphs:
            if p.text.strip():
                text.append(p.text.strip())
    except Exception as e:
        st.error(f"DOCX error: {e}")
    return "\n".join(text)


# -------------------------------------------------------------
# SAFE WORD-BASED CHUNKING
# -------------------------------------------------------------
def chunk_text(text, max_length=1500, overlap=150):
    """Chunk text without breaking words."""
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if len(current_chunk) + len(word) + 1 > max_length:
            chunks.append(current_chunk.strip())

            # Keep last <overlap> words
            overlap_words = current_chunk.split()[-overlap:]
            current_chunk = " ".join(overlap_words) + " "

        current_chunk += word + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# -------------------------------------------------------------
# BULK EMBEDDING 20x FASTER
# -------------------------------------------------------------
def embed_chunks_bulk(chunks):
    try:
        resp = genai.embed_content(
            model=EMBED_MODEL,
            content=chunks,
            task_type="retrieval_document"
        )
        return resp["embedding"]
    except Exception as e:
        st.warning(f"Embedding error: {e}")
        return None


def embed(text):
    """Single embed for user questions."""
    try:
        resp = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_query"
        )
        return np.array(resp["embedding"])
    except Exception as e:
        st.warning(f"Embedding error: {e}")
        return None


# -------------------------------------------------------------
# Search
# -------------------------------------------------------------
def search(query, top_k=3):
    if not VECTOR_STORE:
        return []

    q_vec = embed(query)
    if q_vec is None:
        return []

    sims = cosine_similarity([q_vec], VECTOR_STORE)[0]
    idxs = sims.argsort()[::-1][:top_k]
    return [DOC_STORE[i] for i in idxs]


# -------------------------------------------------------------
# Prompt Builder
# -------------------------------------------------------------
def build_prompt(contexts, question):
    ctx = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    return f"""
Answer ONLY using the context and add citations like [1], [2].
If answer not found, say "I don't know".

Context:
{ctx}

Question: {question}

Answer with citations:
"""


# -------------------------------------------------------------
# Sidebar Chat History
# -------------------------------------------------------------
with st.sidebar:
    st.title("💬 Chat History")

    if st.button("🧹 Clear Chat History"):
        st.session_state.chat_history = []
        st.success("History cleared.")

    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            role = "🟦 You" if msg["role"] == "user" else "🟧 Assistant"
            st.markdown(f"**{role}:** {msg['text'][:800]}")
    else:
        st.info("No messages yet.")


# -------------------------------------------------------------
# MAIN UI
# -------------------------------------------------------------
st.title("📘 Knowledge Base Agent")

# Developer Header
st.markdown("""
<div style='padding: 12px; background-color: #F2F4F7; border-radius: 10px; margin-bottom: 20px;'>
    <h3 style='margin-bottom: 0; color:#333;'>👩‍💻 Developed by <b>Suchithra G</b></h3>
    <p style='font-size: 16px; color: #555; margin-top: 5px;'>
        🏆 AI Agent Development Challenge – Knowledge Base Agent
    </p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])

# -------------------------------------------------------------
# Upload + Process
# -------------------------------------------------------------
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    ext = uploaded.name.split(".")[-1].lower()

    # Extract text
    if ext == "pdf":
        full_text = extract_text_from_pdf(path)
    elif ext == "docx":
        full_text = extract_text_from_docx(path)
    else:
        full_text = open(path, "r", encoding="utf-8", errors="ignore").read()

    if not full_text.strip():
        st.error("No text found in document.")
    else:
        chunks = chunk_text(full_text)

        with st.spinner("⚡ Fast Embedding Document..."):
            embeddings = embed_chunks_bulk(chunks)

            if embeddings:
                for chunk, emb in zip(chunks, embeddings):
                    DOC_STORE.append(chunk)
                    VECTOR_STORE.append(np.array(emb))

        st.success("Document added to knowledge base!")


# -------------------------------------------------------------
# Question Input
# -------------------------------------------------------------
st.markdown("---")

query = st.text_input("Ask a question:")

if st.button("Get Answer") and query.strip():

    st.session_state.chat_history.append({"role": "user", "text": query})

    results = search(query, top_k=3)

    if not results:
        answer = "I don't know."
        st.warning(answer)
        st.session_state.chat_history.append({"role": "assistant", "text": answer})
    else:
        prompt = build_prompt(results, query)
        resp = GEN_MODEL.generate_content(prompt)
        answer = resp.text

        st.subheader("Answer")
        st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "text": answer})

        st.subheader("Context Used")
        for i, chunk in enumerate(results, start=1):
            st.markdown(f"### [{i}]")
            st.write(chunk[:500] + ("..." if len(chunk) > 500 else ""))
