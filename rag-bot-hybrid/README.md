# Hybrid Retrieval System (POC to Production)

Hybrid RAG chatbot with:
- keyword retrieval (Elasticsearch)
- vector retrieval (Qdrant)
- RRF fusion in hybrid mode
- optional cross-encoder reranking
- provider-switchable embeddings and generation (OpenAI, Gemini, Ollama)
- Streamlit UI + online and batch evaluation

## Architecture

### POC profile (default)
- Keyword: Elasticsearch (Docker)
- Vector: Qdrant (Docker)
- Embedding: OpenAI text-embedding-3-small (1536-d)
- Generator: OpenAI gpt-4o-mini
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (local)

### Production profile (config target)
- Keyword: managed Elasticsearch
- Vector: Pinecone (adapter placeholder in config)
- Embedding: OpenAI text-embedding-3-large (3072-d)
- Generator: configurable
- Reranker: Cohere rerank-v3.5 (profile definition)

## Project Structure

### Entry points
- `app.py` - Streamlit app (chat, infra checks, ingestion UI, online eval panel)
- `evaluate.py` - offline batch evaluator

### Core package
- `ragbot/config/settings.py` - config load + runtime profile/provider overrides
- `ragbot/ingestion/ingest.py` - PDF parsing, chunking, indexing, file move to processed
- `ragbot/retrieval/stores.py` - Elasticsearch and Qdrant store adapters
- `ragbot/retrieval/retriever.py` - keyword/vector/hybrid retrieval + RRF merge
- `ragbot/retrieval/reranker.py` - cross-encoder reranking
- `ragbot/llm/models.py` - embedding + generation provider implementations
- `ragbot/evaluation/evaluation.py` - logging + scoring utilities (grounding, hit@k, token F1, exact match)

### Data and ops files
- `config.yml` - retrieval, reranking, ingestion, evaluation, and profile definitions
- `docker-compose.yml` - local Elasticsearch + Qdrant
- `data/` - incoming PDFs
- `data/processed/` - already indexed PDFs
- `eval_data/` - batch eval datasets
- `eval_logs/` - online and batch eval outputs

## Setup

1. Create and activate virtual environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create `.env` manually in this folder.

```env
# Required for default POC embedding + generator
OPENAI_API_KEY=your_key_here

# Optional based on selected generator
GOOGLE_API_KEY=your_key_here
OLLAMA_BASE_URL=http://localhost:11434

# Optional infra overrides
ELASTICSEARCH_URL=http://localhost:9200
QDRANT_URL=http://localhost:6333

# Optional runtime defaults
ENV_MODE=poc
LOG_LEVEL=INFO
APP_LOG_PATH=logs/app.log
```

4. Start local infra.

```bash
docker compose up -d
```

5. Run app.

```bash
streamlit run app.py
```

## How to Use

1. In sidebar, choose profile, embedding provider, and generator provider.
2. In Infrastructure section, verify Elasticsearch and Qdrant are online.
3. Upload PDFs and click Ingest uploaded PDFs.
4. Ask questions in Chat tab.
5. Expand Sources to inspect retrieval origin and rerank scores.

## Retrieval Modes

- `keyword` - BM25-style text search
- `vector` - nearest-neighbor vector search
- `hybrid` - merges keyword + vector with Reciprocal Rank Fusion (RRF)

RRF scoring uses configured `rrf_k` in `config.yml`.

## Evaluation

### Online (inside app)

When `evaluation.enabled=true`, each answer is appended to:
- `eval_logs/interactions.jsonl`

Evaluation panel shows:
- total queries
- avg and p95 latency
- avg grounding overlap
- trend chart + recent rows

### Offline batch

1. Prepare dataset.

```bash
cp eval_data/eval_dataset.template.jsonl eval_data/eval_dataset.jsonl
```

2. Run evaluator.

```bash
python evaluate.py \
	--dataset eval_data/eval_dataset.jsonl \
	--retrieval-mode hybrid
```

3. Output.
- per-query JSONL: `eval_logs/batch_eval_results.jsonl`
- terminal aggregate metrics: latency, grounding overlap, hit@k, token F1

## Configuration Notes

- Main knobs are in `config.yml`:
	- `retrieval.top_k`, `retrieval.rrf_k`
	- `reranking.enabled`, `reranking.top_n`
	- `ingestion.chunk_size`, `ingestion.chunk_overlap`
	- `profiles.poc` and `profiles.production`

- Provider override behavior:
	- Profile gives baseline model/provider values.
	- Sidebar provider choices override profile provider/model/dims at runtime.

- If embedding dimensions change, recreate backing indexes before re-ingestion.

## Troubleshooting

- App imports fail from system Python:
	- Run from project venv: `./.venv/bin/python ...`

- Infra offline in app:
	- Check Docker Desktop running
	- Run `docker compose ps`
	- Verify URLs in `.env`

- PDF ingested but no results:
	- Confirm file moved to `data/processed/`
	- Check `logs/app.log` for ingestion/indexing errors

- Missing package/provider errors in sidebar:
	- Install requirements in active venv
	- Add required API keys in `.env`
