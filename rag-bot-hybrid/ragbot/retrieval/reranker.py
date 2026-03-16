from sentence_transformers import CrossEncoder


class LocalReranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[dict], top_n: int) -> tuple[list[dict], list[float]]:
        if not hits:
            return [], []
        pairs = [(query, hit["text"]) for hit in hits]
        scores = self.model.predict(pairs).tolist()
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [item[0] for item in ranked], [float(item[1]) for item in ranked]
