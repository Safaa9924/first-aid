"""
tfidf_index.py

Builds the TF-IDF index from the semantic chunks CSV and saves it to
data/tfidf_index.pkl — same TfidfVectorizer settings as the notebook
(Stage 7a), just pulled into its own runnable script so it can be re-run
as part of the pipeline (after 01_documents.py, before 05_create_chroma_store.py).

Note: unlike MiniBM25, sklearn's TfidfVectorizer is a library class (not
defined in __main__), so it doesn't hit the "module 'main' has no
attribute ..." pickling problem — as long as sklearn is installed wherever
you load it, pickle.load will work fine.

Run:
    python tfidf_index.py
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

CHUNKS_CSV = "data/first_aid_semantic_chunks_final.csv"
OUTPUT_PATH = "data/tfidf_index.pkl"


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
        dtype="float32"
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


def load_tfidf_index(path=OUTPUT_PATH):
    """Helper for other scripts (e.g. the retrieval / app script) to load
    the saved index back."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["vectorizer"], data["matrix"]


if __name__ == "__main__":

    os.makedirs("data", exist_ok=True)

    chunks_df = pd.read_csv(CHUNKS_CSV)
    texts = chunks_df["chunk_text"].tolist()

    tfidf_vectorizer, tfidf_matrix = build_tfidf_index(texts)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "vectorizer": tfidf_vectorizer,
            "matrix": tfidf_matrix
        }, f)

    print(f"\nSaved: {OUTPUT_PATH}")
