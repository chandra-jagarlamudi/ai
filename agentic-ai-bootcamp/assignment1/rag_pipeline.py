import os
import time
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

# Type checking for vectorstores only used for comparison runs
if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS

# Open string types — any registered provider name is valid.
LLMProvider = str
EmbeddingProvider = str


# ── Provider factory functions ─────────────────────────────────────────────────
# Each function receives a RAGConfig and returns the appropriate LLM / Embeddings.

def _make_openai_llm(cfg: "RAGConfig") -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=cfg.llm_model)

def _make_openai_embeddings(cfg: "RAGConfig") -> Embeddings:
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=cfg.embedding_model)

def _make_gemini_llm(cfg: "RAGConfig") -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=cfg.llm_model)

def _make_gemini_embeddings(cfg: "RAGConfig") -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=cfg.embedding_model)

def _make_ollama_llm(cfg: "RAGConfig") -> BaseChatModel:
    from langchain_ollama import ChatOllama
    return ChatOllama(model=cfg.llm_model, base_url=cfg.ollama_base_url)

def _make_ollama_embeddings(cfg: "RAGConfig") -> Embeddings:
    try:
        from langchain_ollama import OllamaEmbeddings
    except Exception:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore
    return OllamaEmbeddings(model=cfg.embedding_model, base_url=cfg.ollama_base_url)


# ── Provider registry ──────────────────────────────────────────────────────────
# To add a new provider, add one entry here — no other code changes needed.
#
# Each entry must have:
#   default_llm_model       – fallback model name if LLM_MODEL / llm_model_env not set
#   llm_model_env           – env var that overrides the LLM model (e.g. OPENAI_CHAT_MODEL)
#   default_embedding_model – fallback embedding model
#   make_llm                – factory: (RAGConfig) -> BaseChatModel
#   make_embeddings         – factory: (RAGConfig) -> Embeddings
#
# Example — adding Anthropic Claude:
#   "anthropic": {
#       "default_llm_model": "claude-sonnet-4-6",
#       "llm_model_env": "ANTHROPIC_CHAT_MODEL",
#       "default_embedding_model": "text-embedding-3-small",  # use openai embeddings
#       "make_llm": _make_anthropic_llm,
#       "make_embeddings": _make_openai_embeddings,
#   },
PROVIDER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "openai": {
        "default_llm_model": "gpt-4o-mini",
        "llm_model_env": "OPENAI_CHAT_MODEL",
        "default_embedding_model": "text-embedding-3-small",
        "make_llm": _make_openai_llm,
        "make_embeddings": _make_openai_embeddings,
    },
    "gemini": {
        "default_llm_model": "gemini-2.5-flash",
        "llm_model_env": "GEMINI_CHAT_MODEL",
        "default_embedding_model": "gemini-embedding-001",
        "make_llm": _make_gemini_llm,
        "make_embeddings": _make_gemini_embeddings,
    },
    "ollama": {
        "default_llm_model": "gemma",
        "llm_model_env": "OLLAMA_CHAT_MODEL",
        "default_embedding_model": "nomic-embed-text",
        "make_llm": _make_ollama_llm,
        "make_embeddings": _make_ollama_embeddings,
    },
}


# RAGConfig class for the RAG pipeline configuration and configuration for the comparison run
@dataclass(frozen=True)
class RAGConfig:
    data_dir: Path
    vectorstore_dir: Path
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 3
    embedding_provider: EmbeddingProvider = "ollama"
    embedding_model: str = "nomic-embed-text"
    llm_provider: LLMProvider = "ollama"
    llm_model: str = "gemma3:1b"
    ollama_base_url: str = "http://localhost:11434"

    @property
    def vectorstore_meta_path(self) -> Path:
        return self.vectorstore_dir / "meta.json"


# Default configuration for the RAG pipeline
def default_config() -> RAGConfig:
    base = Path(__file__).resolve().parent
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    # Resolve LLM provider and model from registry
    llm_spec = PROVIDER_REGISTRY.get(llm_provider)
    if llm_spec is None:
        logger.warning("Unknown LLM_PROVIDER=%s, falling back to openai", llm_provider)
        llm_provider = "openai"
        llm_spec = PROVIDER_REGISTRY["openai"]
    llm_model = (
        os.getenv(llm_spec["llm_model_env"])
        or os.getenv("LLM_MODEL")
        or llm_spec["default_llm_model"]
    )

    # Resolve embedding provider and model from registry
    emb_spec = PROVIDER_REGISTRY.get(embedding_provider)
    if emb_spec is None:
        logger.warning("Unknown EMBEDDING_PROVIDER=%s, falling back to openai", embedding_provider)
        embedding_provider = "openai"
        emb_spec = PROVIDER_REGISTRY["openai"]
    embedding_model = os.getenv("EMBEDDING_MODEL") or emb_spec["default_embedding_model"]

    # App chat: default to provider-specific vectorstore (e.g. vectorstore_openai)
    vectorstore_dir = os.getenv("VECTORSTORE_DIR") or str(base / f"vectorstore_{llm_provider}")

    cfg = RAGConfig(
        data_dir=Path(os.getenv("DATA_DIR", str(base / "data"))),
        vectorstore_dir=Path(vectorstore_dir),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("TOP_K", "3")),
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    logger.info(
        "Config loaded: llm=%s:%s embeddings=%s:%s vectorstore=%s",
        cfg.llm_provider,
        cfg.llm_model,
        cfg.embedding_provider,
        cfg.embedding_model,
        str(cfg.vectorstore_dir),
    )
    return cfg


# Return a copy of base config with only the LLM provider and model changed (for comparison runs).
def config_for_llm(base: RAGConfig, llm_provider: LLMProvider, llm_model: str) -> RAGConfig:
    """Return a copy of base config with only the LLM provider and model changed (for comparison runs)."""
    return replace(base, llm_provider=llm_provider, llm_model=llm_model)


# Config for the comparison run: one shared vectorstore (e.g. vectorstore_comparison) with OpenAI embeddings so all LLMs run against the same index.
def comparison_config() -> RAGConfig:
    """
    Config for the comparison run: one shared vectorstore (e.g. vectorstore_comparison)
    with OpenAI embeddings so all LLMs run against the same index.
    Uses COMPARISON_EMBEDDING_MODEL (or text-embedding-3-small) so the comparison
    index never uses a non-OpenAI embedding model from EMBEDDING_MODEL.
    """
    base = Path(__file__).resolve().parent
    comparison_dir = os.getenv("COMPARISON_VECTORSTORE_DIR", str(base / "vectorstore_comparison"))
    comparison_embedding = os.getenv("COMPARISON_EMBEDDING_MODEL", "text-embedding-3-small")
    cfg = RAGConfig(
        data_dir=Path(os.getenv("DATA_DIR", str(base / "data"))),
        vectorstore_dir=Path(comparison_dir),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("TOP_K", "3")),
        embedding_provider="openai",
        embedding_model=comparison_embedding,
        llm_provider="openai",
        llm_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    logger.info(
        "Comparison config loaded: embeddings=openai:%s vectorstore=%s",
        cfg.embedding_model,
        str(cfg.vectorstore_dir),
    )
    return cfg


# List the PDF paths in the data directory
def list_pdf_paths(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"])


# Load the documents from the PDFs
def load_documents_from_pdfs(pdf_paths: Iterable[Path]) -> List[Document]:
    docs: List[Document] = []
    for p in pdf_paths:
        loader = PyPDFLoader(str(p))
        docs.extend(loader.load())
    return docs


# Split the documents into chunks
def split_documents(docs: List[Document], *, chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def get_embeddings(cfg: RAGConfig) -> Embeddings:
    spec = PROVIDER_REGISTRY.get(cfg.embedding_provider)
    if spec is None:
        raise ValueError(
            f"Unsupported embedding_provider: {cfg.embedding_provider!r}. "
            f"Register it in PROVIDER_REGISTRY in rag_pipeline.py."
        )
    logger.info("Embeddings selected: %s:%s", cfg.embedding_provider, cfg.embedding_model)
    return spec["make_embeddings"](cfg)


def get_llm(cfg: RAGConfig) -> BaseChatModel:
    spec = PROVIDER_REGISTRY.get(cfg.llm_provider)
    if spec is None:
        raise ValueError(
            f"Unsupported llm_provider: {cfg.llm_provider!r}. "
            f"Register it in PROVIDER_REGISTRY in rag_pipeline.py."
        )
    logger.info("LLM selected: %s:%s", cfg.llm_provider, cfg.llm_model)
    return spec["make_llm"](cfg)


def _vectorstore_meta(cfg: RAGConfig) -> Dict[str, Any]:
    return {
        "embedding_provider": cfg.embedding_provider,
        "embedding_model": cfg.embedding_model,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
    }


def build_and_save_vectorstore(cfg: RAGConfig):
    from langchain_community.vectorstores import FAISS

    logger.info("Vectorstore build started: dir=%s data_dir=%s", str(cfg.vectorstore_dir), str(cfg.data_dir))
    try:
        pdf_paths = list_pdf_paths(cfg.data_dir)
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in {cfg.data_dir}")
        logger.info("Vectorstore build: found_pdfs=%d", len(pdf_paths))

        docs = load_documents_from_pdfs(pdf_paths)
        chunks = split_documents(docs, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        logger.info("Vectorstore build: chunks=%d chunk_size=%d overlap=%d", len(chunks), cfg.chunk_size, cfg.chunk_overlap)

        embeddings = get_embeddings(cfg)
        vs = FAISS.from_documents(chunks, embeddings)
        cfg.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(cfg.vectorstore_dir))
        cfg.vectorstore_meta_path.write_text(json.dumps(_vectorstore_meta(cfg), indent=2), encoding="utf-8")
        logger.info("Vectorstore build succeeded: dir=%s", str(cfg.vectorstore_dir))
        return vs
    except Exception:
        logger.exception("Vectorstore build failed: dir=%s", str(cfg.vectorstore_dir))
        raise


# Load the vectorstore from the filesystem
def load_vectorstore(cfg: RAGConfig):
    from langchain_community.vectorstores import FAISS

    # Check if the vectorstore exists and is valid
    if not cfg.vectorstore_dir.exists():
        logger.debug("Vectorstore missing dir: %s", str(cfg.vectorstore_dir))
        return None
    if not (cfg.vectorstore_dir / "index.faiss").is_file():
        logger.debug("Vectorstore missing index.faiss: %s", str(cfg.vectorstore_dir))
        return None
    if not (cfg.vectorstore_dir / "index.pkl").is_file():
        logger.debug("Vectorstore missing index.pkl: %s", str(cfg.vectorstore_dir))
        return None
    if cfg.vectorstore_meta_path.is_file():
        try:
            meta = json.loads(cfg.vectorstore_meta_path.read_text(encoding="utf-8"))
            expected = _vectorstore_meta(cfg)
            for k, v in expected.items():
                if meta.get(k) != v:
                    logger.warning(
                        "Vectorstore meta mismatch (%s): expected=%s actual=%s. Rebuild required.",
                        k,
                        expected.get(k),
                        meta.get(k),
                    )
                    return None
        except Exception:
            logger.exception("Vectorstore meta read failed: %s", str(cfg.vectorstore_meta_path))
            return None

    embeddings = get_embeddings(cfg)
    try:
        vs = FAISS.load_local(
            str(cfg.vectorstore_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Vectorstore load succeeded: %s", str(cfg.vectorstore_dir))
        return vs
    except Exception:
        logger.exception("Vectorstore load failed: %s", str(cfg.vectorstore_dir))
        return None


# Get the retriever for the vectorstore
def get_retriever(vectorstore, *, top_k: int):
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


# Default prompt for the RAG pipeline
def default_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """Answer the question using only the context below.

If the answer is not present in the context, say: "I don't know based on the provided documents."

Context:
{context}

Question: {question}
"""
    )


# Generate the answer for the question
def rag_answer(
    *,
    question: str,
    vectorstore,
    llm: BaseChatModel,
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
