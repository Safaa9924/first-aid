"""
================================================================================
 STAGE 04 · VECTOR REPRESENTATION
 First Aid Reference Guide (St. John Ambulance Canada) — RAG Pipeline
================================================================================
Builds the three retrieval indexes used for hybrid search:
    - TF-IDF index (lexical, n-grams)
    - BM25 index   (lexical, probabilistic ranking)
    - Sentence-embedding index (semantic, all-MiniLM-L6-v2)

All indexes are pickled/saved to disk for Stage 05 (Chroma store) and
Stage 06 (retrieval) to reuse without recomputing.

Usage:
    python 04_vector_representation.py
================================================================================
"""

import os
import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "data"
CHUNKS_CSV_PATH = os.path.join(DATA_DIR, "first_aid_semantic_chunks_final.csv")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf_index.pkl")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embedding_matrix.npy")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def simple_tokenize(text):
    """Simple tokenizer for BM25."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


# ==================================================================
# Stage 7a — TF-IDF Index
# ==================================================================

def build_tfidf_index(texts):

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
        dtype="float32",
    )

    matrix = vectorizer.fit_transform(texts)

    print("=" * 60)
    print("TF-IDF INDEX SUMMARY")
    print("=" * 60)
    print(f"Documents       : {len(texts)}")
    print(f"Vocabulary Size : {len(vectorizer.vocabulary_):,}")
    print(f"Matrix Shape    : {matrix.shape}")
    print(f"Non-zero Terms  : {matrix.nnz:,}")

    sparsity = (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])) * 100
    print(f"Sparsity        : {sparsity:.2f}%")

    return vectorizer, matrix


# ==================================================================
# Stage 7b — BM25 Index
# ==================================================================

class MiniBM25:

    def __init__(self, tokenized_docs, k1=1.5, b=0.75):

        self.k1 = k1
        self.b = b

        self.docs = tokenized_docs
        self.N = len(tokenized_docs)

        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = np.mean(self.doc_lens)

        # Precompute term frequencies
        self.term_freqs = [Counter(doc) for doc in tokenized_docs]

        # Document frequency
        self.df = Counter()
        for doc in tokenized_docs:
            self.df.update(set(doc))

        # BM25 IDF
        self.idf = {
            term: np.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

        print("=" * 60)
        print("BM25 INDEX SUMMARY")
        print("=" * 60)
        print(f"Documents      : {self.N}")
        print(f"Vocabulary     : {len(self.df):,}")
        print(f"Average Length : {self.avgdl:.1f} words")

    def get_scores(self, query_tokens):

        scores = np.zeros(self.N, dtype=np.float32)

        for term in query_tokens:

            if term not in self.idf:
                continue

            idf = self.idf[term]

            for i, tf_dict in enumerate(self.term_freqs):

                tf = tf_dict.get(term, 0)

                if tf == 0:
                    continue

                denom = (
                    tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl)
                )

                scores[i] += (idf * tf * (self.k1 + 1)) / denom

        return scores


def min_max_normalize(scores):

    scores = np.asarray(scores, dtype=np.float32)

    if scores.size == 0:
        return scores

    lo = scores.min()
    hi = scores.max()

    if hi == lo:
        return np.zeros_like(scores)

    return (scores - lo) / (hi - lo)


# ==================================================================
# Stage 8 — Semantic Embedding Index
# ==================================================================

def build_embedding_index(chunks_df, model_name=EMBEDDING_MODEL_NAME):

    from sentence_transformers import SentenceTransformer

    print("=" * 60)
    print("BUILDING EMBEDDING INDEX")
    print("=" * 60)

    model = SentenceTransformer(model_name)

    texts = chunks_df["chunk_text"].tolist()

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Embedding Model : {model_name}")
    print(f"Documents       : {len(texts)}")
    print(f"Embedding Shape : {embeddings.shape}")

    return model, embeddings


if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 04 · VECTOR REPRESENTATION")
    print("=" * 60)

    chunks_df = pd.read_csv(CHUNKS_CSV_PATH)
    texts = chunks_df["chunk_text"].tolist()

    # ---- TF-IDF ----
    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(texts)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump({"vectorizer": tfidf_vectorizer, "matrix": tfidf_matrix}, f)
    print(f"Saved TF-IDF index -> {TFIDF_PATH}")

    # ---- BM25 ----
    tokenized_docs = [simple_tokenize(t) for t in texts]
    bm25 = MiniBM25(tokenized_docs)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved BM25 index -> {BM25_PATH}")

    # ---- Embeddings ----
    embedding_model, embedding_matrix = build_embedding_index(chunks_df)
    np.save(EMBEDDINGS_PATH, embedding_matrix)
    print(f"Saved embedding matrix -> {EMBEDDINGS_PATH}")

    print("\nDone. Next: run 05_create_chroma_store.py")
