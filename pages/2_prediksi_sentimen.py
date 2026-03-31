import streamlit as st
from utils.loader import load_model
from utils.preprocessing import preprocessing_data
import pandas as pd
import os

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Prediksi Sentimen - Analisis Produk",
    layout="wide"
)

# ==============================================================================
# 2. CUSTOM CSS
# ==============================================================================
def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ==============================================================================
# 3. LOAD MODEL FINAL (SMOTE) & TF-IDF
# ==============================================================================
try:
    # load_model() HARUS mengembalikan model SMOTE
    model, tfidf = load_model()
except:
    st.error("⚠️ Model atau TF-IDF gagal dimuat. Pastikan file model tersedia.")
    st.stop()

# ==============================================================================
# 4. LABEL MAPPING
# ==============================================================================
label_mapping = {
    0: "Negatif",
    1: "Positif"
}

# ==============================================================================
# 5. HEADER & INFO
# ==============================================================================
st.markdown(
    '<div class="blue-header">📝 PREDIKSI SENTIMEN ULASAN</div>',
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#7f8c8d; margin-bottom:25px;'>"
    "Masukkan ulasan pengguna tentang produk pembersih wajah untuk mengetahui sentimen terhadap produk pembersih wajah."
    "</p>",
    unsafe_allow_html=True
)

# col_info1, col_info2 = st.columns(2)

# with col_info1:
#     st.markdown(
#         """
#         <div style="
#             background-color:#E8F2FF;
#             padding:15px;
#             border-radius:10px;
#             font-weight:600;
#             color:#3498DB;
#             font-size:18px;
#         ">
#             🤖 Model: Logistic Regression (SMOTE)
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# with col_info2:
#     st.markdown(
#         """
#         <div style="
#             background-color:#e9f7ef;
#             padding:15px;
#             border-radius:10px;
#             font-weight:600;
#             color:#158237;
#             font-size:18px;
#         ">
#             🎯 Akurasi Model: 78.00%
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

# st.caption(
#     "Catatan: Model telah dilatih menggunakan teknik SMOTE untuk mengatasi ketidakseimbangan data latih."
# )

# st.markdown("---")

# ==============================================================================
# 6. INPUT TEXT
# ==============================================================================
text = st.text_area(
    "Masukkan ulasan produk pembersih wajah:",
    placeholder="Contoh: Produk ini sangat bagus dan membuat wajah terasa segar.",
    height=150
)

# ==============================================================================
# 7. PROSES PREDIKSI
# ==============================================================================
if st.button("🔍 Mulai Prediksi"):

    if text.strip() == "":
        st.warning("⚠️ Harap masukkan teks ulasan terlebih dahulu.")
        st.stop()

    with st.spinner("Sedang memproses ulasan..."):

        # -----------------------------
        # PREPROCESSING
        # -----------------------------
        hasil = preprocessing_data(text)

        # TF-IDF transform
        vec = tfidf.transform([hasil["stemming_data"]])

        # -----------------------------
        # PREDIKSI
        # -----------------------------
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        sentimen = label_mapping[pred]

    st.markdown("---")

    # ==============================================================================
    # 8. HASIL UTAMA
    # ==============================================================================
    st.subheader("📊 Hasil Prediksi Sentimen")

    if sentimen == "Positif":
        st.success(f"### 😊 Sentimen Ulasan: **{sentimen}**")
    else:
        st.error(f"### 😠 Sentimen Ulasan: **{sentimen}**")

    # ==============================================================================
    # 9. PROBABILITAS MODEL
    # ==============================================================================
    st.subheader("📈 Probabilitas Keyakinan Model")

    prob_df = pd.DataFrame({
        "Sentimen": ["Negatif", "Positif"],
        "Probabilitas": proba
    })

    st.dataframe(prob_df, use_container_width=True)
    st.bar_chart(prob_df.set_index("Sentimen"))

    # ==============================================================================
    # 10. KESIMPULAN OTOMATIS
    # ==============================================================================
    st.markdown("---")
    st.subheader("📝 Kesimpulan Analisis")

    if sentimen == "Positif":
        st.success(
            "Hasil prediksi menunjukkan bahwa ulasan memiliki sentimen **positif**, "
            "yang mengindikasikan kepuasan pengguna terhadap produk pembersih wajah."
        )
    else:
        st.error(
            "Hasil prediksi menunjukkan bahwa ulasan memiliki sentimen **negatif**, "
            "yang mengindikasikan adanya ketidakpuasan pengguna terhadap produk pembersih wajah."
        )

    # ==============================================================================
    # 11. DETAIL PREPROCESSING
    # ==============================================================================
    with st.expander("🔎 Lihat Detail Tahapan Preprocessing"):
        st.write("**Teks Asli:**", text)
        st.write("**1. Cleaning:**", hasil["cleaning"])
        st.write("**2. Case Folding:**", hasil["case_folding"])
        st.write("**3. Normalisasi:**", hasil["normalisasi"])
        st.write("**4. Tokenizing:**", hasil["tokenize"])
        st.write("**5. Stopword Removal:**", hasil["stopword_removal"])
        st.write("**6. Stemming (Final):**", hasil["stemming_data"])
