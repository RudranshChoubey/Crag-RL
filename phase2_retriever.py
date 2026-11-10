"""Phase 2 - dense retrieval over the English Wikipedia index in ChromaDB.

Every query is embedded with the SAME multilingual MiniLM encoder the index was built with.
(The old reward called collection.query(query_texts=...), which silently used ChromaDB's default
English model instead - different vector space, meaningless distances.)
"""
from typing import List

import numpy as np

from crag_utils import CHROMA_DIR, COLLECTION, ENCODER_NAME, get_device


class KnowledgeRetriever:
    def __init__(self, db_path: str = CHROMA_DIR, collection_name: str = COLLECTION, encoder=None, device=None):
        import chromadb

        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(ENCODER_NAME, device=device or get_device())
        self.encoder = encoder
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.encoder.encode(list(texts), normalize_embeddings=True, batch_size=64,
                                              show_progress_bar=False), dtype=np.float32)

    def search(self, queries: List[str], top_k: int = 3, query_emb: np.ndarray = None):
        """Batch search. Returns, per query, a list of {text, title, score} (score = cosine similarity)."""
        if query_emb is None:
            query_emb = self.encode(queries)
        res = self.collection.query(query_embeddings=query_emb.tolist(), n_results=top_k,
                                    include=["documents", "metadatas", "distances"])
        out = []
        for docs, metas, dists in zip(res["documents"], res["metadatas"], res["distances"]):
            metas = metas or [{}] * len(docs)
            out.append([{"text": t, "title": (m or {}).get("title", ""), "score": float(1.0 - d)}
                        for t, m, d in zip(docs, metas, dists)])
        return out

    def retrieve_with_scores(self, query: str, top_k: int = 3):
        return self.search([query], top_k)[0]

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Backwards-compatible: list of passage strings."""
        return [p["text"] for p in self.retrieve_with_scores(query, top_k)]

    def count(self):
        return self.collection.count()


if __name__ == "__main__":
    r = KnowledgeRetriever()
    print(f"[*] Collection has {r.count()} passages")
    for q in ["Mount Everest elevation height", "capital of Russia", "swallow chewing gum digestion",
              "Luke Kuechly tackles"]:
        print(f"\nQuery: {q}")
        for i, p in enumerate(r.retrieve_with_scores(q, top_k=3), 1):
            print(f"  [{i}] ({p['score']:.3f}) {p['text'][:150]}...")
