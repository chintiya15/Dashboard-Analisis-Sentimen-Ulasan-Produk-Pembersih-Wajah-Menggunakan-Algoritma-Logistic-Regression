import streamlit as st
import pandas as pd
from utils.preprocessing import full_preprocessing
from utils.deteksi_kolom import detect_text_column
import os

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Preprocessing CSV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# CUSTOM CSS
# =============================
def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =============================
# HEADER
# =============================
st.markdown(
    '<div class="blue-header">📂 PREPROCESSING DATA CSV</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center; color: #7f8c8d; margin-bottom:30px;'>
    Halaman ini digunakan untuk melakukan preprocessing otomatis terhadap data teks ulasan sebelum dilakukan analisis sentimen menggunakan <b>Logistic Regression</b>.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =============================
# INFO FILE
# =============================
st.info("""
📌 **Ketentuan File CSV**
- Format file: **.csv**
- File harus memiliki **minimal satu kolom teks**
- Kolom teks akan dideteksi otomatis oleh sistem
- Kolom lain akan diabaikan
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

    df_raw = pd.read_csv(file)

    # =============================
    # DETEKSI KOLOM TEKS (UTILS)
    # =============================
    text_col = detect_text_column(df_raw)

    if text_col is None:
        st.error("❌ Sistem tidak menemukan kolom teks ulasan yang valid.")
        st.stop()

    st.success(f"✅ Kolom teks terdeteksi: **{text_col}**")

    # =============================
    # AMBIL KOLOM TEKS
    # =============================
    df = df_raw[[text_col]].copy()
    df.rename(columns={text_col: "text_asli"}, inplace=True)

    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    st.subheader("⚙️ Proses Preprocessing")

    progress = st.progress(0)
    status_text = st.empty()
    hasil_preprocessing = []

    for i, text in enumerate(df["text_asli"].astype(str)):
        status_text.text(f"Memproses data ke-{i+1} dari {len(df)}")
        hasil = full_preprocessing(text)
        hasil_preprocessing.append(hasil)
        progress.progress((i + 1) / len(df))

    status_text.empty()
    progress.empty()

    # =============================
    # SIMPAN HASIL PREPROCESSING LENGKAP
    # =============================
    df["cleaning"] = [h["cleaning"] for h in hasil_preprocessing]
    df["case_folding"] = [h["case_folding"] for h in hasil_preprocessing]
    df["normalisasi"] = [h["normalisasi"] for h in hasil_preprocessing]
    df["tokenize"] = [", ".join(h["tokenize"]) for h in hasil_preprocessing]
    df["stopword_removal"] = [", ".join(h["stopword_removal"]) for h in hasil_preprocessing]
    df["stemming_data"] = [h["stemming_data"] for h in hasil_preprocessing]

    st.success("✅ Preprocessing selesai!")

    st.markdown("---")

    st.subheader("📋 Preview Hasil Preprocessing")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")

    csv_result = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV Hasil Preprocessing",
        data=csv_result,
        file_name="hasil_preprocessing.csv",
        mime="text/csv"
    )
