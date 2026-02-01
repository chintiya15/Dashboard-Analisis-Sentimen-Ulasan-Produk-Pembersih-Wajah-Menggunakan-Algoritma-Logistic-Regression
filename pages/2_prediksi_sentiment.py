import streamlit as st
from utils.loader import load_model
from utils.preprocessing import full_preprocessing
import pandas as pd
import os

# ==============================================================================
# 1. KONFIGURASI HALAMAN (Wajib di bagian paling atas)
# ==============================================================================
st.set_page_config(
    page_title="Prediksi Sentimen - Analisis Produk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CUSTOM CSS (MENGHAPUS TOP BAR HITAM & STYLING WARNA)
# ==============================================================================
def load_css():
    css_path = os.path.join("assets", "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ==============================================================================
# 3. LOAD MODEL & MAPPING
# ==============================================================================
model, tfidf = load_model()

label_mapping = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

# ==============================================================================
# 4. CONTENT - HEADER & INFO
# ==============================================================================
st.markdown('<div class="blue-header">📝 PREDIKSI SENTIMEN ULASAN</div>', unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color: #7f8c8d; margin-bottom:30px;'>Ketahui sentimen pelanggan terhadap produk Anda dalam hitungan detik</p>", unsafe_allow_html=True)

col_info1, col_info2 = st.columns(2)
with col_info1:
    st.markdown(
        """
        <div style="
            background-color:#E8F2FF;
            padding:15px;
            border-radius:10px;
            font-weight:600;
            color:#3498DB;
            text-align:left;
            font-size:18px;
        ">
            🤖 Model: Logistic Regression
        </div>
        """,
        unsafe_allow_html=True
    )
    # st.info("🤖 **Model:** Logistic Regression")
with col_info2:
    st.markdown(
        """
        <div style="
            background-color:#e9f7ef;
            padding:15px;
            border-radius:10px;
            font-weight:600;
            color:#158237;
            text-align:left;
            font-size:18px;
        ">
            🎯 Akurasi Model: 72.00%
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")

# ==============================================================================
# 5. CONTENT - INPUT AREA
# ==============================================================================
text = st.text_area(
    "Masukkan ulasan produk pembersih wajah:",
    placeholder="Contoh: pembersih wajah ini sangat lembut dan tidak membuat kulit kering",
    height=150
)

# ==============================================================================
# 6. LOGIC PREDIKSI
# ==============================================================================
if st.button("🔍 Mulai Prediksi"):
    if text.strip() == "":
        st.warning("⚠️ Harap masukkan teks ulasan terlebih dahulu.")
    else:
        with st.spinner('Sedang memproses teks...'):
            # Preprocessing
            hasil = full_preprocessing(text)
            vec = tfidf.transform([hasil['stemming_data']])

            # Prediksi
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            sentimen = label_mapping[pred]

            st.markdown("---")

            # --- HASIL UTAMA ---
            st.subheader("📊 Hasil Analisis")
            
            if sentimen == "Positif":
                st.success(f"### 😊 Sentimen: **{sentimen}**")
            elif sentimen == "Netral":
                st.info(f"### 😐 Sentimen: **{sentimen}**")
            else:
                st.error(f"### 😠 Sentimen: **{sentimen}**")

            # --- GRAFIK PROBABILITAS ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📈 Probabilitas Keyakinan Model")
            
            prob_df = pd.DataFrame({
                "Sentimen": ["Negatif", "Netral", "Positif"],
                "Probabilitas": proba
            })
            
            st.bar_chart(prob_df.set_index("Sentimen"))

            # --- DETAIL PREPROCESSING ---
            with st.expander("🔎 Lihat Detail Tahapan Pembersihan Teks"):
                st.write("**Teks Asli:**", text)
                st.write("**1. Cleaning:**", hasil["cleaning"])
                st.write("**2. Case Folding:**", hasil["case_folding"])
                st.write("**3. Normalisasi:**", hasil["normalisasi"])
                st.write("**4. Tokenizing:**", hasil["tokenize"])
                st.write("**5. Stopword Removal:**", hasil["stopword_removal"])
                st.write("**6. Stemming (Final):**", hasil["stemming_data"])