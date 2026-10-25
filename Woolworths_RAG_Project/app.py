import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from rag import RAGPipeline
from config import CHUNKS_FILE, OUTPUT_DIR, EMBEDDING_MODEL, OLLAMA_MODEL

st.set_page_config(
    page_title="Woolworths FY25 Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1250px; }
    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #f4f8f2 0%, #ffffff 55%, #eef6ec 100%);
        border: 1px solid #dfe9dc;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.25rem; letter-spacing: -0.03em; }
    .hero p { margin: .55rem 0 0 0; color: #59635a; font-size: 1.05rem; }
    .metric-card {
        padding: 1rem 1.1rem;
        border: 1px solid #e4e8e3;
        border-radius: 15px;
        background: #fff;
        min-height: 92px;
    }
    .metric-label { color: #687268; font-size: .82rem; }
    .metric-value { font-size: 1.2rem; font-weight: 700; margin-top: .25rem; }
    .source-card {
        padding: .85rem 1rem;
        border-left: 4px solid #5d8f54;
        background: #f7faf6;
        border-radius: 0 10px 10px 0;
        margin: .45rem 0;
    }
    .small-muted { color: #707970; font-size: .82rem; }
    .answer-box {
        padding: 1.2rem 1.35rem;
        border-radius: 16px;
        border: 1px solid #e2e8e0;
        background: #fbfcfb;
    }
    div[data-testid="stSidebar"] { border-right: 1px solid #e7ebe6; }
    .stButton > button { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline():
    return RAGPipeline(5)


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def run_query(question):
    try:
        return get_pipeline().answer(question), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>📊 Woolworths FY25 Intelligence</h1>
    <p>Ask natural-language questions about the supplied Woolworths FY25 report data. The system retrieves relevant evidence first, then generates a grounded answer.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Search settings")
    top_k = st.slider("Sources to retrieve", 3, 10, 5)
    st.caption("The current pipeline uses BM25 + dense retrieval with Reciprocal Rank Fusion.")

    st.divider()
    st.header("💡 Try a question")
    examples = [
        "Net cash provided by operating activities?",
        "Total net debt including lease liabilities",
        "How did net debt change across the five-year period?",
        "What was Woolworths Group's revenue in 2025?",
    ]
    for i, q in enumerate(examples):
        if st.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.question = q

    st.divider()
    st.header("🔎 System status")
    metric_card("Knowledge base", CHUNKS_FILE.name)
    metric_card("Embedding model", "MiniLM semantic search")
    metric_card("LLM runtime", OLLAMA_MODEL)

# -----------------------------
# Query area
# -----------------------------
question = st.text_area(
    "Ask your question",
    key="question",
    height=100,
    placeholder="Example: What was total net debt in FY2025?",
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    ask = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.session_state.question = ""
    st.rerun()

# -----------------------------
# Results
# -----------------------------
if ask and question.strip():
    with st.spinner("Searching the report and generating a grounded answer…"):
        result, error = run_query(question.strip())

    if error:
        st.error("The website could not complete the query.")
        st.code(error)
        st.info(
            "If this is the first run, make sure the dense embeddings have been built and Ollama is running. "
            "See README_BEGINNER.md for the setup commands."
        )
    else:
        st.session_state.last_result = result

result = st.session_state.get("last_result")

if result:
    st.divider()
    st.subheader("Answer")
    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

    st.markdown("### 📚 Supporting evidence")
    st.caption("These are the passages retrieved by the hybrid search. Check the source before relying on an answer.")

    for i, item in enumerate(result["hybrid"], start=1):
        page = item.get("page", "N/A")
        chunk_id = item.get("chunk_id", "N/A")
        score = item.get("score")
        score_text = f" • RRF score: {score:.4f}" if isinstance(score, (int, float)) else ""
        with st.expander(f"Source {i} — Page {page} — {chunk_id}{score_text}"):
            st.write(item.get("text", ""))

    with st.expander("🧪 See retrieval details"):
        tab1, tab2 = st.tabs(["BM25", "Dense"])
        with tab1:
            for item in result["bm25"]:
                st.markdown(f"**Rank {item.get('rank')} — Page {item.get('page')}**")
                st.write(item.get("text", ""))
                st.divider()
        with tab2:
            for item in result["dense"]:
                st.markdown(f"**Rank {item.get('rank')} — Page {item.get('page')}**")
                st.write(item.get("text", ""))
                st.divider()
else:
    st.info("Enter a question above or choose one of the example questions in the sidebar.")

st.divider()
st.caption("Woolworths FY25 RAG prototype • Retrieval first, generation second • Always verify important figures against the source report.")
