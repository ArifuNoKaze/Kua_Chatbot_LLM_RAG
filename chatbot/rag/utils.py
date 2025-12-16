import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import ollama
import os
import re
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
TEXT_PATH = os.path.join(DATA_DIR, "texts.npy")
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")

# HARUS sama dengan model yg dipakai saat build index
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# load
embedder = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_PATH)
texts = np.load(TEXT_PATH, allow_pickle=True)
embeddings = np.load(EMB_PATH)

# normalize safety
emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
emb_norms[emb_norms == 0] = 1.0
embeddings = embeddings / emb_norms

KEYWORDS = [
    "rujuk", "cerai", "perceraian", "pengadilan", "PA", "iddah",
    "syarat", "persyaratan", "dokumen", "berkas",
    "wali", "nikah", "pendaftaran"
]

GREETINGS = [
    "halo", "hai", "hello", "hi",
    "assalamualaikum", "assalamu'alaikum",
    "selamat pagi", "selamat siang",
    "selamat sore", "selamat malam"
]

def is_greeting(text):
    t = text.lower().strip()
    for g in GREETINGS:
        if g in t:
            return True
    return False

def greeting_response():
    responses = [
        "Assalamualaikum. Selamat datang di layanan informasi KUA. Ada yang bisa kami bantu?",
        "Halo, selamat datang di layanan resmi KUA. Silakan ajukan pertanyaan Anda.",
        "Selamat datang di layanan informasi KUA. Ada informasi yang ingin Anda tanyakan?",
        "Assalamualaikum. Kami siap membantu informasi seputar layanan KUA."
    ]
    return np.random.choice(responses)




def keyword_boost(chunk):
    score = 0
    c = chunk.lower()
    for kw in KEYWORDS:
        if kw in c:
            score += 1
    return score


def retrieve_context(query, top_k=8, faiss_k=30, alpha=0.85, beta=0.15, min_sim=0.15):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype("float32")
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

    D, I = index.search(q_emb, faiss_k)
    idxs = [int(i) for i in I[0] if i != -1]

    ranked = []
    for idx in idxs:
        chunk = texts[idx]
        sim = float(np.dot(q_emb[0], embeddings[idx]))
        kw = keyword_boost(chunk)
        combined = alpha * sim + beta * (kw / (1 + kw))
        ranked.append((idx, chunk, sim, kw, combined))

    ranked = sorted(ranked, key=lambda x: x[4], reverse=True)

    # === HARD THRESHOLD untuk mencegah halusinasi ===
    best_sim = ranked[0][2]
    if best_sim < min_sim:
        print("\n❌ No relevant context found (SIM too low)")
        return None

    selected = ranked[:top_k]

    print("\n=== RETRIEVED (Combined sim + keyword) ===")
    for i, (idx, t, sim, kw, comb) in enumerate(selected):
        print(f"{i+1}. sim={sim:.3f} kw={kw} comb={comb:.4f} -> {t[:120]}...")

    return "\n\n".join([r[1] for r in selected])



# ---------------------------------------------------------
# SAFE JSON PARSER untuk memastikan output format JSON
# ---------------------------------------------------------
def extract_json(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass

    return {"answer": "Jawaban tidak ditemukan dalam konteks."}



# ---------------------------------------------------------
#         LLM ANSWER GENERATOR (phi3 aman)
# ---------------------------------------------------------
def ask_llm(question):

    if is_greeting(question):
        return greeting_response()

    context = retrieve_context(question)

    # Jika tidak ditemukan konteks relevan → langsung JSON
    if context is None:
        return "Jawaban tidak ditemukan dalam konteks."
    

    CLOSINGS = [
    "Apakah ada yang bisa saya bantu lagi?",
    "Ada pertanyaan lain yang ingin Anda tanyakan?",
    "Masih butuh informasi lainnya?",
    "Silakan bertanya kembali jika ada yang belum jelas.",
    "Perlu bantuan lain terkait layanan KUA?"
    ]


    prompt = f"""
Anda adalah Chatbot KUA yang hanya menjawab berdasarkan "konteks" yang diberikan oleh sistem melalui RAG.

ATURAN WAJIB (HARUS DIIKUTI SECARA KETAT):

Jika jawabannya ADA di konteks → jawab secara singkat, jelas, dan sesuai konteks.

Jika jawabannya TIDAK ADA di konteks → jawab:
"Jawaban tidak ditemukan dalam konteks."

Jangan membuat asumsi, jangan mengarang, jangan menambahkan informasi tambahan.

Jangan menggunakan pengetahuan luar (tidak ada data selain konteks).

Jika konteks berisi beberapa jawaban yang mirip:

pilih jawaban yang paling relevan dengan pertanyaan pengguna.

Jika konteks tidak sesuai atau tidak relevan → tetap jawab "Jawaban tidak ditemukan dalam konteks."

Tetap sopan, formal, dan singkat seperti staf KUA.

Di akhir jawaban, tambahkan satu kalimat penutup layanan dengan pilihan kalimat secara bervariasi.
Contoh: “Ada yang bisa saya bantu lagi?”, “Perlu informasi tambahan?”, atau kalimat penutup serupa.
Gunakan variasi berbeda setiap respons.


Format jawaban:
Jangan menyebut bahwa Anda adalah AI.

Tujuan:
Memberikan informasi resmi mengenai prosedur pendaftaran nikah, rujuk, dokumen, biaya, jadwal, peraturan, dan ketentuan terkait layanan KUA, hanya berdasarkan konteks RAG.

{{
  "answer": "<jawaban>"
}}

Konteks:
{context}

Pertanyaan:
{question}

Berikan jawaban JSON:
"""

    response = ollama.generate(
        model="qwen2.5:3b",
        prompt=prompt,
        options={"num_ctx": 4096, "max_tokens": 300}
    )

    raw_output = response["response"]
    parsed = extract_json(raw_output)

    closing = np.random.choice(CLOSINGS)
    final_answer = parsed["answer"] + " " + closing
    return final_answer





# ---------------------------------------------------------
#               RAG EVALUATION SECTION
# ---------------------------------------------------------
def rag_evaluate(testset_path, sim_threshold=0.6):

    """
    Evaluasi performa RAG + LLM.
    
    Parameters:
    - testset_path: path ke JSON test set
    - embedder: sentence embedding model
    - index: FAISS index
    - texts: list dokumen yang diretrieval
    - embeddings: list embedding dokumen
    - sim_threshold: threshold cosine similarity untuk dianggap "benar"
    - ask_llm: fungsi callable untuk mendapatkan jawaban LLM
    """

    with open(testset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    total = len(dataset)
    top1 = 0
    top3 = 0
    cosine_scores = []

    correct_llm = 0
    wrong_llm = 0

    for item in dataset:
        q = item["question"]
        expected = item["expected_answer"]

        q_emb = embedder.encode([q], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

        D, I = index.search(q_emb, 10)
        retrieved_idxs = [int(i) for i in I[0] if i != -1][:3]

        retrieved_texts = [texts[i] for i in retrieved_idxs]

        exp_emb = embedder.encode([expected], convert_to_numpy=True)
        exp_emb = exp_emb / (np.linalg.norm(exp_emb, axis=1, keepdims=True) + 1e-12)

        sims = []
        for i in retrieved_idxs:
            sim = float(np.dot(exp_emb[0], embeddings[i]))
            sims.append(sim)

        if len(sims) > 0:
            if sims[0] >= sim_threshold:
                top1 += 1
            if any(s >= sim_threshold for s in sims[:3]):
                top3 += 1

        llm_out = ask_llm(q)

        llm_emb = embedder.encode([llm_out], convert_to_numpy=True)
        llm_emb = llm_emb / (np.linalg.norm(llm_emb, axis=1, keepdims=True) + 1e-12)

        sim_llm = float(np.dot(llm_emb[0], exp_emb[0]))
        cosine_scores.append(sim_llm)

        if sim_llm >= sim_threshold:
            correct_llm += 1
        else:
            wrong_llm += 1

    # Statistik tambahan
    cosine_scores_arr = np.array(cosine_scores)
    avg_cosine = float(np.mean(cosine_scores_arr)) if cosine_scores else 0.0
    min_cosine = float(np.min(cosine_scores_arr)) if cosine_scores else 0.0
    max_cosine = float(np.max(cosine_scores_arr)) if cosine_scores else 0.0
    median_cosine = float(np.median(cosine_scores_arr)) if cosine_scores else 0.0
    # Tambahkan di bagian akhir sebelum return/print result
    tp = correct_llm
    fn = wrong_llm
    # Asumsi tidak ada false positives eksplisit
    fp = 0  

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    accuracy = correct_llm / total if total > 0 else 0.0

    result = {
        "total_questions": total,
        "retrieval_top1_accuracy": top1 / total if total > 0 else 0.0,
        "retrieval_top3_accuracy": top3 / total if total > 0 else 0.0,
        "avg_cosine_similarity": avg_cosine,
        "min_cosine_similarity": min_cosine,
        "max_cosine_similarity": max_cosine,
        "median_cosine_similarity": median_cosine,
        "llm_accuracy_by_threshold": accuracy,
        "llm_correct_count": correct_llm,
        "llm_wrong_count": wrong_llm,
        "llm_precision": precision,
        "llm_recall": recall,
        "llm_f1_score": f1_score
    }

    print("\n===== RAG EVALUATION SUMMARY =====")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Visualisasi dengan matplotlib
    # Visualisasi dengan matplotlib
    plt.figure(figsize=(14, 6))

    # Histogram cosine similarity
    plt.subplot(1, 3, 1)
    plt.hist(cosine_scores_arr, bins=10, color='skyblue', edgecolor='black')
    plt.title('Distribusi Cosine Similarity LLM vs Expected')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Jumlah Pertanyaan')

    # Bar chart retrieval Top-1 vs Top-3
    plt.subplot(1, 3, 2)
    retrieval_metrics = ["Retrieval Top-1", "Retrieval Top-3"]
    retrieval_values = [
        result["retrieval_top1_accuracy"],
        result["retrieval_top3_accuracy"]
    ]
    plt.bar(retrieval_metrics, retrieval_values, color=['orange', 'green'])
    plt.ylim(0, 1)
    plt.title('Retrieval Accuracy')
    for i, v in enumerate(retrieval_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    # Bar chart LLM metrics
    plt.subplot(1, 3, 3)
    llm_metrics = ["LLM Accuracy", "LLM Precision", "LLM Recall", "LLM F1"]
    llm_values = [
        result["llm_accuracy_by_threshold"],
        result["llm_precision"],
        result["llm_recall"],
        result["llm_f1_score"]
    ]
    colors = ['blue', 'purple', 'red', 'cyan']
    plt.bar(llm_metrics, llm_values, color=colors)
    plt.ylim(0, 1)
    plt.title('LLM Evaluation Metrics')
    for i, v in enumerate(llm_values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.tight_layout()
    plt.show()


    return result
