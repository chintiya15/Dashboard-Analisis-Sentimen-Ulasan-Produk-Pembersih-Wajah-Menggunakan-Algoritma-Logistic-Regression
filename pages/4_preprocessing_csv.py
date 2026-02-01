import streamlit as st
import pandas as pd
from utils.loader import load_model
from utils.preprocessing import full_preprocessing
import os

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Preprocessing Data CSV",
    layout="wide"
)

# =============================
# CUSTOM CSS (KONSISTEN DENGAN FOTO KE-2)
# =============================
def load_css():
    css_path = os.path.join("assets", "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =============================
# LOAD MODEL & TF-IDF
# =============================
model, tfidf = load_model()

label_mapping = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

# =============================
# HEADER
# =============================
st.markdown('<div class="blue-header">📂 PREPROCESSING & ANALISIS SENTIMEN DATA CSV</div>', unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color: #7f8c8d; margin-bottom:30px;'>Halaman ini digunakan untuk melakukan preprocessing data ulasandan analisis sentimen secara <b>batch</b> menggunakan <b>Logistic Regression</b></p>", unsafe_allow_html=True)

st.markdown("---")

# =============================
# INFO FORMAT CSV
# =============================
st.info("""
📌 **Ketentuan File CSV**
- Format file: **.csv**
- Wajib memiliki kolom: **full_text**
- Kolom lain (jika ada) akan diabaikan
""")

# =============================
# UPLOAD FILE
# =============================
file = st.file_uploader(
    "📤 Upload File CSV",
    type=["csv"]
)

# =============================
# PROSES FILE
# =============================
if file is not None:
    df = pd.read_csv(file)

    if "full_text" not in df.columns:
        st.error("❌ Kolom **full_text** tidak ditemukan dalam file CSV.")
    else:
        st.success(f"✅ File berhasil dimuat. Total data: **{len(df)}**")

        st.markdown("---")

        # =============================
        # PREPROCESSING
        # =============================
        st.subheader("⚙️ Proses Preprocessing & Prediksi Sentimen")

        progress = st.progress(0)
        hasil_preprocessing = []

        # Gunakan status untuk memberi tahu user proses sedang berjalan
        status_text = st.empty()
        
        for i, text in enumerate(df["full_text"].astype(str)):
            status_text.text(f"Memproses data ke-{i+1} dari {len(df)}...")
            hasil = full_preprocessing(text)
            hasil_preprocessing.append(hasil)
            progress.progress((i + 1) / len(df))

        status_text.empty()
        progress.empty()

        # =============================
        # SIMPAN HASIL PREPROCESSING
        # =============================
        df["cleaning"] = [h["cleaning"] for h in hasil_preprocessing]
        df["case_folding"] = [h["case_folding"] for h in hasil_preprocessing]
        df["normalisasi"] = [h["normalisasi"] for h in hasil_preprocessing]
        df["tokenize"] = [", ".join(h["tokenize"]) for h in hasil_preprocessing]
        df["stopword_removal"] = [", ".join(h["stopword_removal"]) for h in hasil_preprocessing]
        df["stemming_data"] = [h["stemming_data"] for h in hasil_preprocessing]

        # =============================
        # VEKTORISASI & PREDIKSI
        # =============================
        vec = tfidf.transform(df["stemming_data"])
        df["label"] = model.predict(vec)
        df["sentimen"] = df["label"].map(label_mapping)

        st.success("✅ Preprocessing dan prediksi sentimen selesai")

        st.markdown("---")

        # =============================
        # TAMPILKAN HASIL RINGKAS
        # =============================
        st.subheader("📋 Hasil Analisis Sentimen (Ringkas)")
        st.dataframe(
            df[["full_text", "sentimen"]].head(20),
            use_container_width=True
        )

        st.markdown("---")

        # =============================
        # DISTRIBUSI SENTIMEN
        # =============================
        st.subheader("📊 Distribusi Sentimen")
        distribusi = df["sentimen"].value_counts()
        st.bar_chart(distribusi)

        st.markdown("---")

        # =============================
        # DETAIL PREPROCESSING
        # =============================
        with st.expander("🔎 Lihat Detail Preprocessing Lengkap"):
            st.dataframe(
                df[[
                    "full_text",
                    "cleaning",
                    "case_folding",
                    "normalisasi",
                    "tokenize",
                    "stopword_removal",
                    "stemming_data",
                    "sentimen"
                ]].head(10),
                use_container_width=True
            )

        st.markdown("---")

        # =============================
        # DOWNLOAD HASIL
        # =============================
        csv_result = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Hasil Preprocessing & Sentimen",
            data=csv_result,
            file_name="hasil_preprocessing_sentimen.csv",
            mime="text/csv"
        )