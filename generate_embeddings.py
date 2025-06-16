import json
import os
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Step 1: Load your book text
with open("typescript_book.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Step 2: Split into chunks (adjust chunk size and overlap if needed)
def split_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

chunks = split_text(text)

# Step 3: Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Generate embeddings
embeddings = model.encode(chunks, show_progress_bar=True)

# Step 5: Save to JSON
data = [{"text": chunk, "embedding": emb.tolist()} for chunk, emb in zip(chunks, embeddings)]

with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"✅ Generated {len(data)} embeddings and saved to embeddings.json")
