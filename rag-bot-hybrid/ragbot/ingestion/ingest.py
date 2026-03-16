import hashlib
import os
import shutil

class _SimpleTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []

        chunks = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += step

        return chunks


def _doc_id(source: str, page_num: int, chunk_num: int, text: str) -> str:
    raw = f"{source}:{page_num}:{chunk_num}:{text[:64]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def ingest_pdf(
    file_path: str,
    embedding_cfg: dict,
    ingestion_cfg: dict,
    elastic_index: str,
    qdrant_collection: str,
) -> int:
    from pypdf import PdfReader

    from ragbot.llm.models import get_embedding
    from ragbot.retrieval.stores import ElasticKeywordStore, QdrantVectorStore

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=ingestion_cfg["chunk_size"],
            chunk_overlap=ingestion_cfg["chunk_overlap"],
        )
    except ModuleNotFoundError:
        splitter = _SimpleTextSplitter(
            chunk_size=ingestion_cfg["chunk_size"],
            chunk_overlap=ingestion_cfg["chunk_overlap"],
        )

    keyword_store = ElasticKeywordStore(index_name=elastic_index)
    vector_store = QdrantVectorStore(collection_name=qdrant_collection, dims=embedding_cfg["dims"])
    keyword_store.ensure_index()
    vector_store.ensure_collection()

    reader = PdfReader(file_path)
    source = os.path.basename(file_path)
    docs = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        chunks = splitter.split_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk, embedding_cfg)
            doc_id = _doc_id(source, page_idx + 1, chunk_idx + 1, chunk)
            docs.append(
                {
                    "doc_id": doc_id,
                    "text": chunk,
                    "vector": embedding,
                    "metadata": {
                        "source": source,
                        "page": page_idx + 1,
                        "chunk": chunk_idx + 1,
                    },
                }
            )

    keyword_store.bulk_upsert(docs)
    vector_store.upsert_points(docs)

    processed_dir = os.path.join(os.path.dirname(file_path), "processed")
    os.makedirs(processed_dir, exist_ok=True)
    shutil.move(file_path, os.path.join(processed_dir, source))

    return len(docs)
