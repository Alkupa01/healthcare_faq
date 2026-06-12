"""
==============================================================================
  Retrieval-based Healthcare QA Chatbot
  TAHAP 3: Interface GUI Streamlit menggunakan Model Fine-tuned IndoBERT
==============================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = "faq_dataset.csv"
MODEL_PATH   = "indobert_finetuned" 
CONFIDENCE_THRESHOLD = 0.45

@st.cache_data
def muat_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df.dropna(subset=["disease", "topic", "question", "answer"])

@st.cache_resource
def muat_model(path: str) -> SentenceTransformer:
    if not os.path.exists(path):
        st.error(f"Folder model '{path}' tidak ditemukan. Jalankan model_training.py terlebih dahulu!")
        st.stop()
    return SentenceTransformer(path)

@st.cache_data
def hitung_kb_embeddings(_model, daftar_pertanyaan: list) -> np.ndarray:
    return _model.encode(daftar_pertanyaan, convert_to_numpy=True, show_progress_bar=False)

def render_sidebar(df: pd.DataFrame):
    """
    Fungsi untuk menampilkan daftar penyakit unik dari dataset langsung pada sidebar
    tanpa menggunakan expander (list terbuka secara permanen).
    """
    st.sidebar.markdown("## 🏥 Cakupan Sistem")
    st.sidebar.write("Daftar penyakit yang terdaftar di dalam *Knowledge Base*:")
    
    # Ambil list unik penyakit, bersihkan spasi, dan urutkan secara alfabetis (A-Z)
    daftar_penyakit = sorted(df["disease"].str.title().unique())
    
    st.sidebar.markdown(f"**Total Penyakit Terdaftar:** `{len(daftar_penyakit)}`")
    st.sidebar.write("") # Memberi jarak spasi sedikit
    
    # Menampilkan langsung berupa list teks terbuka secara permanen
    for i, penyakit in enumerate(daftar_penyakit, start=1):
        st.sidebar.markdown(f"{i}. {penyakit}")
            
    st.sidebar.divider()
    st.sidebar.caption(
        "💡 *Tips: Gunakan daftar di atas sebagai acuan pengujian sistem "
        "agar terhindar dari pertanyaan di luar domain dataset.*"
    )

def main():
    st.set_page_config(page_title="Healthcare QA IndoBERT", page_icon="🏥", layout="centered")
    
    st.title("🏥 Healthcare QA System")
    st.markdown("Sistem pencarian informasi kesehatan berbasis **Fine-tuned IndoBERT**.")
    st.divider()
    
    # Validasi keberadaan file dataset sebelum dimuat
    if not os.path.exists(DATASET_PATH):
        st.error(f"❌ File dataset tidak ditemukan di: {DATASET_PATH}")
        st.stop()
        
    df = muat_dataset(DATASET_PATH)
    model = muat_model(MODEL_PATH)
    
    # Panggil fungsi sidebar dengan menyuplai dataframe yang sudah dimuat
    render_sidebar(df)
    
    pertanyaan_kb = df["question"].tolist()
    embeddings_kb = hitung_kb_embeddings(model, pertanyaan_kb)
    
    # UI: Berdampingan secara horizontal antara Input dan Tombol
    col_input, col_btn = st.columns([4, 1], vertical_alignment="bottom")
    with col_input:
        query_user = st.text_input("✏️ Masukkan Pertanyaan Kesehatan Anda:", placeholder="Contoh: bagaimana cara mencegah diabetes?")
    with col_btn:
        tombol_cari = st.button("🔍 Cari", type="primary", use_container_width=True)
        
    st.write("")
    top_k = st.select_slider("⚙️ Jumlah dokumen alternatif (Top-K):", options=[1, 3, 5], value=3)
    
    if tombol_cari and query_user.strip():
        with st.spinner("Menghitung kedekatan semantik vektor IndoBERT..."):
            embedding_query = model.encode([query_user.strip()], convert_to_numpy=True)
            skor_sim = cosine_similarity(embedding_query, embeddings_kb)[0]
            
            top_k_idx = np.argsort(skor_sim)[::-1][:top_k]
            
            st.divider()
            
            # Tampilkan Hasil Utama (Peringkat 1)
            idx_utama = top_k_idx[0]
            skor_utama = skor_sim[idx_utama]
            penyakit_utama = df.iloc[idx_utama]['disease']
            
            if skor_utama < CONFIDENCE_THRESHOLD:
                st.error(f"⚠️ Maaf, sistem tidak menemukan dokumen medis yang relevan (Skor Kepercayaan: {skor_utama:.2%}).")
            else:
                st.success(f"✅ Penyakit Terdeteksi: **{penyakit_utama.upper()}** ({df.iloc[idx_utama]['topic'].upper()})")
                
                col_q, col_s = st.columns([3, 1])
                with col_q:
                    st.markdown(f"**Pertanyaan Terdekat:**\n*{df.iloc[idx_utama]['question']}*")
                with col_s:
                    st.markdown(f"**Confidence Score**\n### {skor_utama:.2%}")
                    
                st.markdown("### 💬 Jawaban Medis:")
                st.info(df.iloc[idx_utama]['answer'])
                
                # Tampilkan Dokumen Alternatif (Top-K) di bawahnya secara melebar
                if top_k > 1:
                    # Filter: Hanya ambil indeks alternatif yang penyakitnya SAMA dengan penyakit utama
                    alternatif_relevan = [idx for idx in top_k_idx[1:] if df.iloc[idx]['disease'] == penyakit_utama]
                    
                    # Hanya buat expander jika masih ada dokumen alternatif setelah difilter
                    if alternatif_relevan:
                        with st.expander("🔍 Lihat Dokumen Relevan Lainnya (Top-K)"):
                            # Gunakan enumerate dengan start=2 karena ini peringkat ke-2 dan seterusnya
                            for rank, idx in enumerate(alternatif_relevan, start=2):
                                st.markdown(f"**#{rank} - {df.iloc[idx]['disease'].title()} ({df.iloc[idx]['topic'].title()})** | Skor: {skor_sim[idx]:.2%}")
                                st.markdown(f"> *{df.iloc[idx]['question']}*")
                                st.markdown(f"Jawaban: {df.iloc[idx]['answer']}")
                                st.divider()

    elif tombol_cari:
        st.warning("Kolom pertanyaan tidak boleh kosong!")

if __name__ == "__main__":
    main()