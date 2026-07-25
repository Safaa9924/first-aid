"""
bm25_utils.py

MiniBM25 class + tokenizer, pulled out of the notebook into their own module.

Why this file exists:
Pickle doesn't save a class's code — only a reference to where it lives
(module.ClassName). When you build `bm25` inside the notebook, its class
lives in `__main__` (or the notebook's kernel), so `bm25_index.pkl` remembers
it as `__main__.MiniBM25`. When a *different* script (e.g. your RAG app)
tries to unpickle that file, Python looks for `MiniBM25` inside ITS OWN
`__main__` and doesn't find it -> "module 'main' has no attribute 'MiniBM25'".

Fix: define MiniBM25 once, here, and import it everywhere (the script that
builds + pickles the index, AND the script that loads it later).
"""

import re
import numpy as np
from collections import Counter


def simple_tokenize(text):
    """Simple tokenizer for BM25."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


class MiniBM25:

    def __init__(self, tokenized_docs, k1=1.5, b=0.75):

        self.k1 = k1
        self.b = b

        self.docs = tokenized_docs
        self.N = len(tokenized_docs)

        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = np.mean(self.doc_lens)

        # Precompute TF (much faster)
        self.term_freqs = [Counter(doc) for doc in tokenized_docs]

        # Document Frequency
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
