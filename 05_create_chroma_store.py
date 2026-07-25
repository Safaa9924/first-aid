"""
================================================================================
 STAGE 05 · CREATE CHROMA VECTOR STORE
 First Aid Reference Guide (St. John Ambulance Canada) — RAG Pipeline
================================================================================
Loads the chunk table + embedding matrix produced by Stage 04 and writes
them into a persistent Chroma collection, used by Stage 06 for semantic
retrieval at query time.

Usage:
    python 05_create_chroma_store.py
================================================================================
"""

import os

import numpy as np
import pandas as pd
import chromadb

DATA_DIR = "data"
CHUNKS_CSV_PATH = os.path.join(DATA_DIR, "first_aid_semantic_chunks_final.csv")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embedding_matrix.npy")

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "first_aid_rag"


def create_chroma_store(chunks_df, embeddings, chroma_path=CHROMA_PATH,
                         collection_name=COLLECTION_NAME):

    client = chromadb.PersistentClient(path=chroma_path)

    print("=" * 60)
    print("CHROMA DATABASE")
    print("=" * 60)
    print(f"Database Path : {chroma_path}")

    collection = client.get_or_create_collection(name=collection_name)
    print(f"Collection    : {collection_name}")

    # Remove previous data so the store always matches the latest chunk table
    existing = collection.count()

    if existing > 0:
        print(f"Existing Documents : {existing}")
        client.delete_collection(collection_name)
        collection = client.get_or_create_collection(name=collection_name)
        print("Old collection removed.")

    collection.add(
        ids=chunks_df["chunk_id"].astype(str).tolist(),
        documents=chunks_df["chunk_text"].tolist(),
        embeddings=embeddings.tolist(),
        metadatas=[
            {"chunk_id": str(row.chunk_id)}
            for row in chunks_df.itertuples()
        ],
    )

    print("=" * 60)
    print("CHROMA STORE CREATED")
    print("=" * 60)
    print(f"Stored Documents : {collection.count()}")

    return client, collection


if __name__ == "__main__":

    print("=" * 60)
    print("STAGE 05 · CREATE CHROMA VECTOR STORE")
    print("=" * 60)

    chunks_df = pd.read_csv(CHUNKS_CSV_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    create_chroma_store(chunks_df, embeddings)

    print("\nDatabase Ready.")
    print("Done. Next: run streamlit_app.py")
