"""
================================================================================
 STAGE 06 · RETRIEVE CONTEXT
 First Aid Reference Guide (St. John Ambulance Canada) — RAG Pipeline
================================================================================
Everything needed to turn a raw user question into a grounded context
package for the LLM:

    language detection -> translation -> query expansion ->
    hybrid retrieval (TF-IDF + BM25 + embeddings) ->
    cross-encoder reranking -> context packaging

This module is imported by both `streamlit_app.py` and `07_prompting.py`.
It performs no file I/O of its own — the caller loads the indexes (see
`load_indexes`) and passes them in.
================================================================================
"""

import os
import re
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"
CHUNKS_CSV_PATH = os.path.join(DATA_DIR, "first_aid_semantic_chunks_final.csv")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf_index.py")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.py")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embedding_matrix.npy")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"

# ==================================================================
# Query expansion dictionary (domain-specific first-aid synonyms)
# ==================================================================

QUERY_EXPANSION = {
    "burn": ["thermal burn", "chemical burn", "critical burn", "burn dressing", "cool water"],
    "fracture": ["splint", "immobilization", "broken bone"],
    "stroke": ["FAST", "facial drooping", "speech difficulty"],
    "choking": ["back blows", "abdominal thrusts", "airway obstruction"],
}


# ==================================================================
# Small helpers
# ==================================================================

def simple_tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def min_max_normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        return scores
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# ==================================================================
# Index loading (call once, cache in the app layer)
# ==================================================================

def load_indexes():
    """Load the chunk table + TF-IDF / BM25 / embedding indexes built by
    Stage 03/04. Returns a dict ready to feed into the retrieval functions."""

    from sentence_transformers import SentenceTransformer

    chunks_df = pd.read_csv(CHUNKS_CSV_PATH)

    with open(TFIDF_PATH, "rb") as f:
        tfidf_bundle = pickle.load(f)

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    embedding_matrix = np.load(EMBEDDINGS_PATH)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return {
        "chunks_df": chunks_df,
        "tfidf_vectorizer": tfidf_bundle["vectorizer"],
        "tfidf_matrix": tfidf_bundle["matrix"],
        "bm25": bm25,
        "embedding_model": embedding_model,
        "embedding_matrix": embedding_matrix,
    }


# ==================================================================
# Language detection & translation
# ==================================================================

def detect_language(text):
    from langdetect import detect
    try:
        return detect(text)
    except Exception:
        return "unknown"


def translate_to_english(text):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="auto", target="en").translate(text)


def translate_to_arabic(text):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="auto", target="ar").translate(text)


def expand_query(query):
    """Expand the (English) retrieval query with domain-specific keywords."""

    expanded_query = query
    lower_query = query.lower()

    for keyword, synonyms in QUERY_EXPANSION.items():
        if keyword in lower_query:
            expanded_query += " " + " ".join(synonyms)

    return expanded_query


# ==================================================================
# Stage 9 — Retrieval functions
# ==================================================================

def retrieve_top_k_tfidf(query, tfidf_vectorizer, tfidf_matrix, chunks_df, k=40):

    q_vec = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    ranking = np.argsort(scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "TF-IDF"

    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


def retrieve_top_k_bm25(query, bm25, chunks_df, k=40):

    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranking = np.argsort(scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = np.array(scores)[ranking]
    results["retriever"] = "BM25"

    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


def retrieve_top_k_semantic(query, embedding_model, embedding_matrix, chunks_df, k=40):

    query_embedding = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )

    scores = cosine_similarity(query_embedding, embedding_matrix).flatten()

    ranking = np.argsort(scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    results["retriever"] = "Embeddings"

    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


def retrieve_top_k_hybrid(
    query,
    tfidf_vectorizer, tfidf_matrix,
    bm25,
    embedding_model, embedding_matrix,
    chunks_df,
    tfidf_weight=0.1,
    bm25_weight=0.1,
    semantic_weight=0.8,
    k=40,
):

    # TF-IDF score
    q_vec = tfidf_vectorizer.transform([query])
    tfidf_scores = min_max_normalize(cosine_similarity(q_vec, tfidf_matrix).flatten())

    # BM25 score
    bm25_scores = min_max_normalize(bm25.get_scores(simple_tokenize(query)))

    # Semantic score
    query_embedding = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    semantic_scores = min_max_normalize(
        cosine_similarity(query_embedding, embedding_matrix).flatten()
    )

    # Weighted fusion
    fused_scores = (
        tfidf_weight * tfidf_scores
        + bm25_weight * bm25_scores
        + semantic_weight * semantic_scores
    )

    ranking = np.argsort(fused_scores)[::-1][:k]

    results = chunks_df.iloc[ranking].copy()
    results["score"] = fused_scores[ranking]
    results["retriever"] = "Hybrid"

    return results[["retriever", "chunk_id", "score", "chunk_text"]].reset_index(drop=True)


# ==================================================================
# Stage 15 — Cross-Encoder Reranking
# ==================================================================

_reranker_cache = {}


def get_reranker():
    from sentence_transformers import CrossEncoder
    if "reranker" not in _reranker_cache:
        _reranker_cache["reranker"] = CrossEncoder(CROSS_ENCODER_NAME)
    return _reranker_cache["reranker"]


def rerank_candidates(query, candidates_df, top_n=10):

    reranker = get_reranker()

    df = candidates_df.copy().reset_index(drop=True)
    df["original_rank"] = range(1, len(df) + 1)

    pairs = [(query, text) for text in df["chunk_text"]]
    df["rerank_score"] = reranker.predict(pairs)

    df = df.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    df["new_rank"] = range(1, len(df) + 1)

    def movement(old_rank, new_rank):
        if new_rank < old_rank:
            return "Improved"
        elif new_rank > old_rank:
            return "Dropped"
        return "Same"

    df["movement"] = df.apply(
        lambda row: movement(row["original_rank"], row["new_rank"]), axis=1
    )

    return df.head(top_n)


# ==================================================================
# Stage 16 — Context package construction
# ==================================================================

def build_context_package(
    query,
    reranked_df,
    max_context_chunks=8,
    word_budget=1500,
    max_chunk_words=180,
):
    """
    Build the final context package for the LLM.

    - Removes duplicate chunks
    - Limits each chunk length
    - Respects total word budget
    - Produces clean context (no scores or metadata)
    """

    candidates = reranked_df.sort_values("rerank_score", ascending=False).reset_index(drop=True)

    if candidates.empty:
        return {
            "query": query,
            "selected_df": pd.DataFrame(),
            "context_text": "",
            "num_sources": 0,
            "used_words": 0,
        }

    selected_rows = []
    seen_texts = set()
    used_words = 0

    for _, row in candidates.iterrows():

        text = row["chunk_text"].strip()
        normalized = re.sub(r"\s+", " ", text).lower()

        if normalized in seen_texts:
            continue

        words = text.split()

        if len(words) > max_chunk_words:
            text = " ".join(words[:max_chunk_words])

        chunk_words = len(text.split())

        if used_words + chunk_words > word_budget:
            break

        row = row.copy()
        row["chunk_text"] = text

        selected_rows.append(row)
        seen_texts.add(normalized)
        used_words += chunk_words

        if len(selected_rows) >= max_context_chunks:
            break

    selected_df = pd.DataFrame(selected_rows)

    context_text = "\n\n---\n\n".join(selected_df["chunk_text"].tolist()) if not selected_df.empty else ""

    return {
        "query": query,
        "selected_df": selected_df,
        "context_text": context_text,
        "num_sources": len(selected_df),
        "used_words": used_words,
    }


# ==================================================================
# End-to-end convenience wrapper
# ==================================================================

def get_context_for_question(
    user_question,
    indexes,
    tfidf_weight=0.1,
    bm25_weight=0.1,
    semantic_weight=0.8,
    top_k=40,
    top_n_rerank=10,
    max_context_chunks=8,
    word_budget=1500,
    max_chunk_words=180,
):
    """Run the full Stage 06 pipeline for a single user question."""

    language = detect_language(user_question)

    retrieval_query = (
        translate_to_english(user_question) if language == "ar" else user_question
    )

    expanded_query = expand_query(retrieval_query)

    hybrid_results = retrieve_top_k_hybrid(
        query=expanded_query,
        tfidf_vectorizer=indexes["tfidf_vectorizer"],
        tfidf_matrix=indexes["tfidf_matrix"],
        bm25=indexes["bm25"],
        embedding_model=indexes["embedding_model"],
        embedding_matrix=indexes["embedding_matrix"],
        chunks_df=indexes["chunks_df"],
        tfidf_weight=tfidf_weight,
        bm25_weight=bm25_weight,
        semantic_weight=semantic_weight,
        k=top_k,
    )

    reranked = rerank_candidates(expanded_query, hybrid_results, top_n=top_n_rerank)

    context = build_context_package(
        query=expanded_query,
        reranked_df=reranked,
        max_context_chunks=max_context_chunks,
        word_budget=word_budget,
        max_chunk_words=max_chunk_words,
    )

    context["language"] = language
    context["retrieval_query"] = retrieval_query

    return context
