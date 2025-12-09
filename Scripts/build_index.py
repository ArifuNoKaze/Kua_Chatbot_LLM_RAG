import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# Optional: you can replace model string with another embedding model you prefer
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DATASET_PATH = "data/raw/dataset_jawaban_kua.json"
SAVE_DIR = "data/faiss_index"
INDEX_PATH = f"{SAVE_DIR}/index.faiss"
EMB_PATH = f"{SAVE_DIR}/embeddings.npy"
TEXT_PATH = f"{SAVE_DIR}/texts.npy"

def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []
    for item in data:
        category = item.get("category", "umum")
        question = item.get("question", "")
        answer = item.get("answer", "")

        text = (
            f"Kategori: {category}\n"
            f"Pertanyaan: {question}\n"
            f"Jawaban: {answer}"
        )
        result.append(text)

    return result



def build_index():
    print("🔍 Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("📄 Loading dataset JSON...")
    texts = load_dataset()

    print(f"📦 Total data: {len(texts)}")

    # NOTE: If your documents are long, chunk them here (e.g., 300-500 tokens with overlap).
    # Example: produce a list `chunks` derived from each document. For now we assume texts are already chunked.
    # TODO: implement chunking if needed.

    print("🔢 Generating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # Normalize embeddings (so inner-product == cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    print("📦 Building FAISS index (IndexFlatIP for cosine via normalized vectors)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("💾 Saving index & embeddings & texts...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    np.save(EMB_PATH, embeddings)
    np.save(TEXT_PATH, np.array(texts, dtype=object))

    print("✅ Index building completed!")
    print("Index saved:", INDEX_PATH)


if __name__ == "__main__":
    build_index()
