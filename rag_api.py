import os
import json
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai

# Load OpenAI API key from env
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Load chunks and embeddings once at startup
# chunks.json should be a list of {id: string, content: string}
# embeddings.json should be a list of floats arrays corresponding to chunks by index

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f.readlines()]

# Load embeddings from precomputed file, e.g. embeddings.json (list of lists)
with open("embeddings.json", "r", encoding="utf-8") as f:
    embeddings = np.array(json.load(f))  # shape (num_chunks, embedding_dim)

if len(chunks) != embeddings.shape[0]:
    raise ValueError("Chunks and embeddings count mismatch")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def embed_text(text: str) -> np.ndarray:
    """Get embedding vector from OpenAI"""
    response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return np.array(response.data[0].embedding)

def get_top_k_chunks(query_embedding: np.ndarray, k: int = 3):
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-10
    )
    top_k_idx = similarities.argsort()[-k:][::-1]
    return [(chunks[i], similarities[i]) for i in top_k_idx]

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    # Concatenate retrieved chunks' content as context for LLM
    context = "\n\n---\n\n".join(chunk["content"] for chunk, _ in retrieved_chunks)
    prompt = (
        f"Answer the question based ONLY on the following excerpts from the TypeScript book documentation.\n\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"Answer concisely with exact excerpts and cite sources if possible."
    )
    return prompt

@app.get("/search")
async def search(q: str = Query(..., description="Question text")):
    try:
        query_embedding = embed_text(q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    retrieved_chunks = get_top_k_chunks(query_embedding, k=3)

    prompt = build_prompt(q, retrieved_chunks)

    try:
        response = openai.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided documentation excerpts."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation error: {str(e)}")

    source_ids = ", ".join(chunk["id"] for chunk, _ in retrieved_chunks)

    return {"answer": answer, "sources": source_ids}
