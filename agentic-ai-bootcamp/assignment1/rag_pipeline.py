import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

@dataclass(frozen=True)
class RAGConfig:
    data_dir: Path
    vectorstore_dir: Path
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 3
    embedding_model: str = "text-embedding-3-small"


def default_config() -> RAGConfig:
    base = Path(__file__).resolve().parent
    return RAGConfig(
        data_dir=Path(os.getenv("DATA_DIR", str(base / "data"))),
        vectorstore_dir=Path(os.getenv("VECTORSTORE_DIR", str(base / "vectorstore"))),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("TOP_K", "3")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )


def list_pdf_paths(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


def load_documents_from_pdfs(pdf_paths: Iterable[Path]) -> List[Document]:
    docs: List[Document] = []
    for p in pdf_paths:
        loader = PyPDFLoader(str(p))
        docs.extend(loader.load())
    return docs


def split_documents(docs: List[Document], *, chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def get_embeddings(model: str) -> OpenAIEmbeddings:
    # Shared embedding model used by all three variants for retrieval.
    # Requires OPENAI_API_KEY in the environment.
    return OpenAIEmbeddings(model=model)


def build_and_save_vectorstore(cfg: RAGConfig):
    from langchain_community.vectorstores import FAISS

    pdf_paths = list_pdf_paths(cfg.data_dir)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {cfg.data_dir}")

    docs = load_documents_from_pdfs(pdf_paths)
    chunks = split_documents(docs, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    embeddings = get_embeddings(cfg.embedding_model)
    vs = FAISS.from_documents(chunks, embeddings)
    cfg.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(cfg.vectorstore_dir))
    return vs


def load_vectorstore(cfg: RAGConfig):
    from langchain_community.vectorstores import FAISS

    if not cfg.vectorstore_dir.exists():
        return None
    if not (cfg.vectorstore_dir / "index.faiss").is_file():
        return None
    if not (cfg.vectorstore_dir / "index.pkl").is_file():
        return None
    embeddings = get_embeddings(cfg.embedding_model)
    try:
        return FAISS.load_local(
            str(cfg.vectorstore_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def get_retriever(vectorstore, *, top_k: int):
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def default_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """Answer the question using only the context below.

If the answer is not present in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question: {question}
"""
    )


def make_openai_llm() -> ChatOpenAI:
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model)


def rag_answer(
    *,
    question: str,
    vectorstore,
    llm: ChatOpenAI,
    cfg: RAGConfig,
    prompt: Optional[ChatPromptTemplate] = None,
) -> Tuple[str, Dict[str, Any]]:
    retriever = get_retriever(vectorstore, top_k=cfg.top_k)
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    chain = (prompt or default_prompt()) | llm | StrOutputParser()

    t0 = time.perf_counter()
    text = chain.invoke({"context": context, "question": question})
    t1 = time.perf_counter()

    meta: Dict[str, Any] = {
        "time_seconds": round(t1 - t0, 4),
        "top_k": cfg.top_k,
        "sources": sorted({str(d.metadata.get("source", "")) for d in docs if d.metadata}),
    }
    return text, meta

