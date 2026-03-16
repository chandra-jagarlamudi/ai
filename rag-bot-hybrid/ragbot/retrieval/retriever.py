from ragbot.retrieval.stores import ElasticKeywordStore, QdrantVectorStore


class HybridRetriever:
    def __init__(self, index_name: str, collection_name: str, dims: int, top_k: int, rrf_k: int):
        self.keyword_store = ElasticKeywordStore(index_name=index_name)
        self.vector_store = QdrantVectorStore(collection_name=collection_name, dims=dims)
        self.top_k = top_k
        self.rrf_k = rrf_k

    def ensure_stores(self):
        self.keyword_store.ensure_index()
        self.vector_store.ensure_collection()

    def search(self, query: str, query_vector: list[float], mode: str) -> list[dict]:
        normalized_mode = (mode or "").strip().lower()

        if normalized_mode == "keyword":
            return self.keyword_store.search(query=query, top_k=self.top_k)
        if normalized_mode == "vector":
            return self.vector_store.search(query_vector=query_vector, top_k=self.top_k)
        if normalized_mode == "hybrid":
            return self._hybrid_search(query=query, query_vector=query_vector)

        raise ValueError(
            f"Unsupported retrieval mode '{mode}'. Expected one of: hybrid, keyword, vector."
        )

    def _hybrid_search(self, query: str, query_vector: list[float]) -> list[dict]:
        keyword_hits = self.keyword_store.search(query=query, top_k=self.top_k)
        vector_hits = self.vector_store.search(query_vector=query_vector, top_k=self.top_k)

        if not keyword_hits and not vector_hits:
            return []

        scores = {}
        docs = {}

        for rank, hit in enumerate(keyword_hits, start=1):
            doc_id = hit["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            docs[doc_id] = hit

        for rank, hit in enumerate(vector_hits, start=1):
            doc_id = hit["doc_id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            if doc_id not in docs:
                docs[doc_id] = hit

        ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[: self.top_k]
        merged = []
        for rank, doc_id in enumerate(ranked_ids, start=1):
            item = dict(docs[doc_id])
            item["score"] = scores[doc_id]
            item["rank"] = rank
            item["retriever"] = "hybrid"
            merged.append(item)
        return merged
