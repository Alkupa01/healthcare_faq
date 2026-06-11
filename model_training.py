import os
import random
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from datasets import Dataset as HFDataset

# =============================================================================
# KONFIGURASI
# =============================================================================
DATASET_PATH = "faq_dataset.csv"  # Menggunakan dataset asli Anda (240 baris)
MODEL_NAME   = "indobenchmark/indobert-base-p1" 
OUTPUT_DIR   = "indobert_finetuned"
BATCH_SIZE   = 16
EPOCHS       = 3  # MNRL sangat aman dan efisien dijalankan dalam 3 epoch
SEED         = 42
TOP_K        = [1, 3, 5]

random.seed(SEED)
np.random.seed(SEED)

# =============================================================================
# LOAD DATASET
# =============================================================================
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["disease", "topic", "question", "answer"])
    print(f"[INFO] Dataset dimuat: {len(df)} baris.")
    return df

# =============================================================================
# GENERATE DATA POSITIF UNTUK CONTRASTIVE LEARNING (MNRL)
# =============================================================================
def buat_training_pairs_mnrl(df: pd.DataFrame):
    """
    MNRL hanya membutuhkan pasangan POSITIF asli (Anchor, Positive).
    Model akan otomatis menganggap baris lain di dalam batch sebagai negatif (pengganggu).
    """
    pairs_data = []
    
    # Kelompokkan berdasarkan penyakit dan topik yang sama
    for (disease, topic), group in df.groupby(["disease", "topic"]):
        questions = group["question"].tolist()
        # Buat kombinasi pasangan positif dari variasi pertanyaan di topik yang sama
        for q1, q2 in combinations(questions, 2):
            pairs_data.append({"text_anchors": q1, "text_positives": q2})
            pairs_data.append({"text_anchors": q2, "text_positives": q1}) # Augmentasi dua arah
            
    random.shuffle(pairs_data)
    print(f"[INFO] Berhasil membentuk {len(pairs_data)} pasangan positif untuk MNRL.")
    return pairs_data

# =============================================================================
# METRIKS EVALUASI PERINGKAT (RECALL & MRR)
# =============================================================================
def hitung_recall_at_k(df_test: pd.DataFrame, df_kb: pd.DataFrame, model: SentenceTransformer, k_values: list) -> dict:
    q_test = df_test["question"].tolist()
    d_test = df_test["disease"].tolist()
    
    q_kb = df_kb["question"].tolist()
    d_kb = df_kb["disease"].tolist()
    
    # Encode pertanyaan uji dan seluruh database pengetahuan (Knowledge Base)
    emb_test = model.encode(q_test, convert_to_numpy=True)
    emb_kb   = model.encode(q_kb, convert_to_numpy=True)
    
    recall_at_k = {k: 0 for k in k_values}

    for i in range(len(q_test)):
        skor_sim = cosine_similarity([emb_test[i]], emb_kb)[0]
        
        # Jika pertanyaan uji ada di KB, abaikan kecocokan dengan dirinya sendiri
        if q_test[i] in q_kb:
            self_idx = q_kb.index(q_test[i])
            skor_sim[self_idx] = -1
            
        ranking_idx = np.argsort(skor_sim)[::-1]

        for k in k_values:
            top_k_penyakit = [d_kb[j] for j in ranking_idx[:k]]
            if d_test[i] in top_k_penyakit:
                recall_at_k[k] += 1

    return {k: v / len(q_test) for k, v in recall_at_k.items()}

def hitung_mrr(df_test: pd.DataFrame, df_kb: pd.DataFrame, model: SentenceTransformer) -> float:
    q_test = df_test["question"].tolist()
    d_test = df_test["disease"].tolist()
    
    q_kb = df_kb["question"].tolist()
    d_kb = df_kb["disease"].tolist()
    
    emb_test = model.encode(q_test, convert_to_numpy=True)
    emb_kb   = model.encode(q_kb, convert_to_numpy=True)
    
    reciprocal_ranks = []

    for i in range(len(q_test)):
        skor_sim = cosine_similarity([emb_test[i]], emb_kb)[0]
        
        if q_test[i] in q_kb:
            self_idx = q_kb.index(q_test[i])
            skor_sim[self_idx] = -1
            
        ranking_idx = np.argsort(skor_sim)[::-1]

        for rank, idx in enumerate(ranking_idx, start=1):
            if d_kb[idx] == d_test[i]:
                reciprocal_ranks.append(1 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks))

# =============================================================================
# PIPELINE UTAMA
# =============================================================================
def main():
    df = load_dataset(DATASET_PATH)
    
    # Pisahkan 15% penyakit/pertanyaan secara utuh untuk data uji evaluasi peringkat
    df_train_raw, df_test_raw = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["disease"])
    
    # Bentuk pasangan latih dari potongan df_train_raw
    train_pairs = buat_training_pairs_mnrl(df_train_raw)
    
    print(f"[INFO] Memuat pre-trained model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    # Inisialisasi MultipleNegativesRankingLoss
    train_loss = MultipleNegativesRankingLoss(model)
    
    # Konfigurasi argumen trainer yang dioptimalkan untuk MNRL
    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        seed=SEED,
        save_strategy="no",  # Mencegah I/O error / crash harddisk di Windows
        logging_steps=5
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=HFDataset.from_list(train_pairs),
        loss=train_loss,
    )
    
    print("[INFO] Memulai Fine-Tuning IndoBERT dengan MNRL...")
    trainer.train()
    
    print(f"[INFO] Menyimpan model akhir ke folder '{OUTPUT_DIR}'...")
    model.save_pretrained(OUTPUT_DIR)
    print(f"[SUKSES] Model berhasil disimpan.")
    
    # Jalankan evaluasi peringkat murni
    print("\n" + "=" * 60)
    print("      HASIL EVALUASI PERINGKAT MODEL AKHIR (TEST SET)")
    print("=" * 60)
    recall = hitung_recall_at_k(df_test_raw, df, model, TOP_K)
    for k, val in recall.items():
        print(f"  Recall@{k:<2}               : {val:.4f}")
    mrr = hitung_mrr(df_test_raw, df, model)
    print(f"  Mean Reciprocal Rank     : {mrr:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()