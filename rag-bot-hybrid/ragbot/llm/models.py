import os
from typing import List

from openai import OpenAI


_EMBEDDER_CACHE = {}


def _load_local_embedder(model_name: str):
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


def get_embedding(text: str, embedding_cfg: dict) -> List[float]:
    provider = embedding_cfg["provider"]
    model = embedding_cfg["model"]

    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    if provider == "huggingface_local":
        embedder = _load_local_embedder(model)
        vector = embedder.encode(text, normalize_embeddings=True)
        return vector.tolist()

    raise ValueError(f"Unsupported embedding provider: {provider}")


def generate_answer(query: str, context: str, generator_cfg: dict, history: list[dict] | None = None) -> str:
    provider = generator_cfg["provider"]
    model = generator_cfg["model"]

    history = history or []
    system_prompt = (
        "You are a grounded RAG assistant. "
        "Answer only from provided context. "
        "If the answer is not in context, say you do not know."
    )

    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""

    if provider == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model_client = genai.GenerativeModel(model)
        conversation = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        prompt = (
            f"System: {system_prompt}\n"
            f"History:\n{conversation}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        resp = model_client.generate_content(prompt)
        return resp.text or ""

    if provider == "ollama":
        import ollama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        client = ollama.Client(host=base_url)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
        resp = client.chat(model=model, messages=messages)
        return resp.message.content

    raise ValueError(f"Unsupported generator provider: {provider}")
