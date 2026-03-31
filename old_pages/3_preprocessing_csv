import streamlit as st
import pandas as pd
from utils.preprocessing import preprocessing_data
from utils.deteksi_kolom import detect_text_column
import os

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Preprocessing CSV",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ==============================================================================
# 2. HEADER
# ==============================================================================
st.markdown('<div class="blue-header">📂 PREPROCESSING DATA CSV</div>', unsafe_allow_html=True)

st.markdown(
    """
    <p style='text-align:center; color: #7f8c8d; margin-bottom:30px;'>
    Halaman ini digunakan untuk melakukan preprocessing otomatis terhadap data teks ulasan sebelum dilakukan analisis sentimen menggunakan <b>Logistic Regression</b>.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==============================================================================
# 3. KETENTUAN FILE (Sesuai Pengujian)
# ==============================================================================
st.info("""
📌 **Ketentuan File CSV**
- Format file wajib: **.csv**
- Ukuran maksimal file: **200MB**
- File harus memiliki **minimal satu kolom teks** ulasan
- Kolom teks akan dideteksi otomatis oleh sistem
""")

# ==============================================================================
# 4. UPLOAD FILE
# ==============================================================================
file = st.file_uploader(
    "📤 Upload atau Drag and Drop File CSV",
    type=["csv"],
    help="Limit 200MB per file - CSV"
)

# ==============================================================================
# 5. LOGIKA PROSES OTOMATIS
# ==============================================================================
if file is not None:
    # Validasi Ukuran File secara manual untuk notifikasi custom
    if file.size > 200 * 1024 * 1024:
        st.error("❌ Ukuran maksimal adalah 200MB.")
    else:
        try:
            df_raw = pd.read_csv(file)
            
            # Deteksi Kolom Teks
            text_col = detect_text_column(df_raw)

            if text_col is None:
                # Notifikasi sesuai hasil pengujian Anda
                st.error("❌ Sistem tidak menemukan kolom teks ulasan yang valid.")
            else:
                st.success(f"✅ File berhasil dimuat. Kolom teks terdeteksi: **{text_col}**")
                
                # Menampilkan Preview Tabel Data Asli
                st.subheader("📋 Preview Data Asli")
                df = df_raw[[text_col]].copy()
                df.rename(columns={text_col: "text_asli"}, inplace=True)
                st.dataframe(df.head(10), use_container_width=True)

                st.markdown("---")
                
                # JALAN OTOMATIS TANPA BUTTON
                st.subheader("⚙️ Proses Preprocessing...")
                
                progress = st.progress(0)
                status_text = st.empty()
                hasil_preprocessing = []
                total_data = len(df)

                # Mulai iterasi pemrosesan
                for i, text in enumerate(df["text_asli"].astype(str)):
                    status_text.text(f"Memproses data ke-{i+1} dari {total_data}")
                    
                    # Memanggil fungsi preprocessing
                    hasil = preprocessing_data(text)
                    hasil_preprocessing.append(hasil)
                    
                    # Update progress bar
                    progress.progress((i + 1) / total_data)

                status_text.empty()
                progress.empty()

                # Menyusun DataFrame Hasil
                df["cleaning"] = [h["cleaning"] for h in hasil_preprocessing]
                df["case_folding"] = [h["case_folding"] for h in hasil_preprocessing]
                df["normalisasi"] = [h["normalisasi"] for h in hasil_preprocessing]
                df["tokenize"] = [", ".join(h["tokenize"]) for h in hasil_preprocessing]
                df["stopword_removal"] = [", ".join(h["stopword_removal"]) for h in hasil_preprocessing]
                df["stemming_data"] = [h["stemming_data"] for h in hasil_preprocessing]

                st.success("✅ Preprocessing selesai!")

                # Menampilkan Tabel Hasil Preprocessing Lengkap
                st.subheader("📋 Tabel Hasil Preprocessing")
                st.dataframe(df, use_container_width=True)

                st.markdown("---")

                # Tombol Download Hasil
                csv_result = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download CSV Hasil Preprocessing",
                    data=csv_result,
                    file_name="hasil_preprocessing_lengkap.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses isi file: {e}")