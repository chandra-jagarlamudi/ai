import os
from typing import Iterable

from elasticsearch import Elasticsearch, NotFoundError, helpers
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class ElasticKeywordStore:
    def __init__(self, index_name: str):
        self.es = Elasticsearch(os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
        self.index_name = index_name

    def ensure_index(self):
        if self.es.indices.exists(index=self.index_name):
            return
        mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "metadata": {
                        "properties": {
                            "source": {"type": "keyword"},
                            "page": {"type": "integer"},
                            "chunk": {"type": "integer"},
                        }
                    },
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=mapping)

    def bulk_upsert(self, docs: Iterable[dict]):
        actions = []
        for doc in docs:
            actions.append(
                {
                    "_index": self.index_name,
                    "_id": doc["doc_id"],
                    "_source": {
                        "doc_id": doc["doc_id"],
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                    },
                }
            )
        if actions:
            helpers.bulk(self.es, actions)
            self.es.indices.refresh(index=self.index_name)

    def search(self, query: str, top_k: int) -> list[dict]:
        try:
            response = self.es.search(
                index=self.index_name,
                query={"match": {"text": query}},
                size=top_k,
            )
        except NotFoundError:
            return []

        hits = []
        for rank, hit in enumerate(response["hits"]["hits"]):
            src = hit["_source"]
            hits.append(
                {
                    "doc_id": src["doc_id"],
                    "text": src["text"],
                    "metadata": src["metadata"],
                    "score": float(hit["_score"]),
                    "rank": rank + 1,
                    "retriever": "keyword",
                }
            )
        return hits


class QdrantVectorStore:
    def __init__(self, collection_name: str, dims: int):
        self.collection_name = collection_name
        self.dims = dims
        self.client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    def ensure_collection(self):
        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dims, distance=Distance.COSINE),
        )

    def upsert_points(self, points: Iterable[dict]):
        payload = []
        for point in points:
            payload.append(
                PointStruct(
                    id=point["doc_id"],
                    vector=point["vector"],
                    payload={
                        "text": point["text"],
                        "metadata": point["metadata"],
                    },
                )
            )
        if payload:
            self.client.upsert(collection_name=self.collection_name, points=payload)

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        hits = []
        for rank, hit in enumerate(results):
            payload = hit.payload or {}
            hits.append(
                {
                    "doc_id": str(hit.id),
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {}),
                    "score": float(hit.score),
                    "rank": rank + 1,
                    "retriever": "vector",
                }
            )
        return hits
