import os
import json
import re

# Directory where the markdown files are (change if needed)
MARKDOWN_DIR = "./typescript-book"

# Output file
OUTPUT_FILE = "chunks.json"

# Parameters
CHUNK_SIZE = 1000  # Approximate characters per chunk

def split_markdown_into_chunks(file_path, chunk_size=CHUNK_SIZE):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Optional: Clean up or remove code blocks if needed
    # text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def create_chunks_json():
    result = []
    for root, _, files in os.walk(MARKDOWN_DIR):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, MARKDOWN_DIR)
                chunks = split_markdown_into_chunks(full_path)

                for i, chunk in enumerate(chunks):
                    result.append({
                        "id": f"{rel_path}#{i}",
                        "content": chunk
                    })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for entry in result:
            json.dump(entry, out)
            out.write("\n")

    print(f"✅ Generated {len(result)} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    create_chunks_json()
