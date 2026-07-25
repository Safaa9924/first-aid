"""
================================================================================
 FIRST AID RAG ASSISTANT — Streamlit Web App
 St. John Ambulance Canada · First Aid Reference Guide, 4th Edition
================================================================================
Attractive, emergency-themed chat UI over the hybrid RAG pipeline
(01_documents -> 07_prompting). Generation runs through OpenRouter.

Your OpenRouter API key is NEVER written in this file. Provide it via:
    - a `.streamlit/secrets.toml` file:   OPENROUTER_API_KEY = "sk-or-..."
    - or an environment variable:         export OPENROUTER_API_KEY=sk-or-...

Run:
    streamlit run streamlit_app.py
================================================================================
"""

import os
import importlib
import streamlit as st

# Modules 06 / 07 start with digits, so we load them with importlib.
retrieve = importlib.import_module("06_retrieve_context")
prompting = importlib.import_module("07_prompting")


# ==================================================================
# Page config + theme
# ==================================================================

st.set_page_config(
    page_title="First Aid RAG Assistant",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --fa-red: #D32F2F;
    --fa-red-dark: #A61E1E;
    --fa-cream: #FFF8F6;
}

.stApp {
    background: linear-gradient(180deg, #FFF8F6 0%, #FFFFFF 220px);
}

/* Header banner */
.fa-header {
    background: linear-gradient(135deg, var(--fa-red) 0%, var(--fa-red-dark) 100%);
    padding: 1.6rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 6px 18px rgba(211, 47, 47, 0.25);
}
.fa-header h1 { margin: 0; font-size: 1.9rem; }
.fa-header p { margin: 0.3rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }

/* Disclaimer banner */
.fa-disclaimer {
    background: #FFF3E0;
    border-left: 6px solid #F57C00;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    font-size: 0.88rem;
    margin-bottom: 1.2rem;
}

/* Confidence badges */
.badge {
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-high { background:#E6F4EA; color:#1E7E34; }
.badge-medium { background:#FFF4E0; color:#B26A00; }
.badge-low { background:#FCE8E8; color:#B00020; }

/* Source card */
.source-card {
    background: #FAFAFA;
    border: 1px solid #EEE;
    border-left: 4px solid var(--fa-red);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
    font-size: 0.85rem;
}

section[data-testid="stSidebar"] {
    background: #FFF8F6;
    border-right: 1px solid #F3D9D9;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==================================================================
# Header
# ==================================================================

st.markdown(
    """
    <div class="fa-header">
        <h1>🚑 First Aid RAG Assistant ⛑️</h1>
        <p>Grounded answers from the St. John Ambulance Canada
        <b>First Aid Reference Guide, 4th Edition</b> — hybrid retrieval +
        cross-encoder reranking + OpenRouter generation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="fa-disclaimer">
    ⚠️ <b>This tool is for educational reference only.</b>
    In a real emergency, call your local emergency number immediately (🚨 911 / 999 / 112).
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================================
# Sidebar — settings
# ==================================================================

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    api_key_present = bool(prompting.get_openrouter_api_key())
    if api_key_present:
        st.success("🔑 OpenRouter API key detected")
    else:
        st.error("🔑 No OpenRouter API key found")
        st.caption(
            "Add `OPENROUTER_API_KEY` to `.streamlit/secrets.toml` "
            "or as an environment variable. The key is never stored in code."
        )

    model = st.selectbox(
        "🧠 OpenRouter model",
        options=[
            "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "openai/gpt-4o-mini",
            "google/gemini-2.0-flash-001",
            "anthropic/claude-3.5-haiku",
        ],
        index=0,
        help="Any chat-completion model available on openrouter.ai",
    )

    st.markdown("### 🎛️ Retrieval weights")
    tfidf_w = st.slider("TF-IDF", 0.0, 1.0, 0.1, 0.05)
    bm25_w = st.slider("BM25", 0.0, 1.0, 0.1, 0.05)
    semantic_w = st.slider("Semantic", 0.0, 1.0, 0.8, 0.05)

    st.markdown("### 🩹 Try asking about")
    st.caption("🔥 Burns · 🦴 Fractures · 🧠 Stroke (FAST) · 😵 Choking\n"
               "❤️ CPR · 🐝 Anaphylaxis · 🥶 Frostbite · 🥵 Heat stroke")

    if st.button("🗑️ Clear conversation"):
        st.session_state.pop("messages", None)
        st.rerun()


# ==================================================================
# Load indexes (cached across reruns)
# ==================================================================

@st.cache_resource(show_spinner="🚑 Loading first-aid knowledge base...")
def get_indexes():
    return retrieve.load_indexes()


try:
    indexes = get_indexes()
    index_ready = True
except Exception as e:
    index_ready = False
    st.error(
        "Could not load the knowledge base. Make sure you have run "
        "01_documents.py -> 05_create_chroma_store.py first.\n\n"
        f"Details: {e}"
    )


# ==================================================================
# Chat state
# ==================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_grounding(grounding):
    if not grounding or grounding["num_sources"] == 0:
        return

    best = grounding["best_score"]
    if best >= 2.5:
        badge_class, label = "badge-high", "🟢 High confidence"
    elif best >= 1.0:
        badge_class, label = "badge-medium", "🟡 Medium confidence"
    else:
        badge_class, label = "badge-low", "🔴 Low confidence"

    st.markdown(f'<span class="badge {badge_class}">{label}</span>', unsafe_allow_html=True)

    with st.expander(f"📚 View {grounding['num_sources']} source excerpt(s)"):
        for src in grounding["sources"]:
            snippet = src["text"][:280].replace("\n", " ")
            st.markdown(
                f"""
                <div class="source-card">
                <b>#{src['rank']} · {src['chunk_id']}</b> — score {src['rerank_score']:.2f}
                ({src['confidence']})<br>{snippet}...
                </div>
                """,
                unsafe_allow_html=True,
            )


for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🚑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("grounding"):
            render_grounding(msg["grounding"])


# ==================================================================
# Chat input
# ==================================================================

question = st.chat_input("🩹 Ask your first-aid question (e.g. 'How do I treat a burn?')")

if question and index_ready:

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🚑"):
        with st.spinner("🔎 Searching the first-aid guide..."):
            context = retrieve.get_context_for_question(
                question,
                indexes,
                tfidf_weight=tfidf_w,
                bm25_weight=bm25_w,
                semantic_weight=semantic_w,
            )

        if context["num_sources"] == 0:
            answer_text = "I couldn't find this information in the retrieved first aid reference. 🚨 If this is an emergency, call your local emergency number now."
            grounding = {"num_sources": 0, "best_score": 0.0, "sources": []}
        else:
            with st.spinner(f"🧠 Generating grounded answer with {model}..."):
                prompting.DEFAULT_MODEL = model
                result = prompting.answer_first_aid_question(
                    question,
                    context,
                    translate_back=retrieve.translate_to_arabic,
                )
            answer_text = result["final_answer"]
            grounding = result["grounding"]

        st.markdown(answer_text)
        render_grounding(grounding)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text, "grounding": grounding}
    )

elif question and not index_ready:
    st.warning("Knowledge base isn't ready yet — see the error above.")
