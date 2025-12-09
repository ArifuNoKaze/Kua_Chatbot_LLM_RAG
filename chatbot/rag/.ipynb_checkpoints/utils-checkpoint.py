import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import ollama
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
TEXT_PATH = os.path.join(DATA_DIR, "texts.npy")
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")


# load model & index
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)
texts = np.load(TEXT_PATH, allow_pickle=True)

KEYWORDS = ["syarat", "persyaratan", "dokumen", "berkas", "wali", "nikah", "pendaftaran"]

def keyword_boost(chunk):
    score = 0
    chunk_low = chunk.lower()
    for kw in KEYWORDS:
        if kw in chunk_low:
            score += 1
    return score

def retrieve_context(query, top_k=5):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    D, I = index.search(q_emb, 30)

    ranked = []
    for idx in I[0]:
        if idx == -1:
            continue
        chunk = texts[idx]
        score = keyword_boost(chunk)
        ranked.append((chunk, score))

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

    selected = [t[0] for t in ranked[:top_k]]

    print("\n=== RETRIEVED (Keyword Reranked) ===")
    for i, t in enumerate(selected):
        print(f"{i+1}. {t[:90]}...")

    return "\n\n".join(selected)

def ask_llm(question):
    context = retrieve_context(question)

    prompt = f"""
Anda adalah Chatbot Informasi Resmi KUA.
Jawaban HARUS hanya berdasarkan konteks berikut.
Jangan menambah informasi baru.

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""

    response = ollama.generate(
        model="phi3",
        prompt=prompt,
        options={"num_ctx": 4096, "max_tokens": 400}
    )

    return response["response"]

###########################################################
# ------------------   EVALUASI RAG   ---------------------
###########################################################

def rag_evaluate(testset_path):
    with open(testset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total = len(dataset)
    top1 = 0
    top3 = 0
    cosine_scores = []

    tp = fp = fn = 0  # for precision/recall

    for item in dataset:
        q = item["question"]
        expected = item["expected_answer"]

        # retrieval test
        q_emb = embedder.encode([q], convert_to_numpy=True)
        D, I = index.search(q_emb, 3)
        retrieved = [texts[i] for i in I[0]]

        if expected in retrieved[0]:
            top1 += 1

        if any(expected in r for r in retrieved):
            top3 += 1

        # LLM answer
        llm_out = ask_llm(q)

        score = util.cos_sim(
            embedder.encode(llm_out),
            embedder.encode(expected)
        ).item()
        cosine_scores.append(score)

        # precision/recall
        if score >= 0.6:
            tp += 1
        else:
            fn += 1

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    result = {
        "total_questions": total,
        "retrieval_top1_accuracy": top1 / total,
        "retrieval_top3_accuracy": top3 / total,
        "avg_cosine_similarity": float(np.mean(cosine_scores)),
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print("\n===== RAG EVALUATION SUMMARY =====")
    for k, v in result.items():
        print(f"{k}: {v}")

    return result
