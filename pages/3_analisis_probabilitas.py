import streamlit as st
import pandas as pd
from utils.loader import load_model
from utils.preprocessing import full_preprocessing
import os

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Analisis Probabilitas Sentimen",
    layout="wide"
)

# =============================
# CUSTOM CSS (WARNA FOTO KE-2)
# =============================
def load_css():
    css_path = os.path.join("assets", "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =============================
# LOAD MODEL
# =============================
model, tfidf = load_model()
label_mapping = {0: "Negatif", 1: "Netral", 2: "Positif"}

# =============================
# HEADER
# =============================
st.markdown('<div class="blue-header">📊 ANALISIS PROBABILITAS SENTIMEN</div>', unsafe_allow_html=True)

st.info("""
Probabilitas menunjukkan tingkat keyakinan model terhadap masing-masing kelas sentimen (Negatif, Netral, Positif).
""")

# =============================
# INPUT TEKS
# =============================
text = st.text_input(
    "Masukkan teks ulasan:",
    value="garnier bikin jerawat"
)

# =============================
# ANALISIS
# =============================
if st.button("📈 Analisis Probabilitas"):
    if text.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        hasil = full_preprocessing(text)
        vec = tfidf.transform([hasil["stemming_data"]])
        prob = model.predict_proba(vec)[0]
        pred = model.predict(vec)[0]
        sentimen_prediksi = label_mapping[pred]

        if sentimen_prediksi == "Positif":
            st.success(f"### 😊 Sentimen Dominan: **{sentimen_prediksi}**")
        elif sentimen_prediksi == "Netral":
            st.info(f"### 😐 Sentimen Dominan: **{sentimen_prediksi}**")
        else:
            st.error(f"### 😠 Sentimen Dominan: **{sentimen_prediksi}**")

        st.markdown("---")

        df_prob = pd.DataFrame({
            "Sentimen": ["Negatif", "Netral", "Positif"],
            "Probabilitas": prob
        })

        col_tabel, col_grafik = st.columns(2)
        with col_tabel:
            st.subheader("📋 Tabel Probabilitas")
            st.dataframe(df_prob, use_container_width=True)
        with col_grafik:
            st.subheader("📊 Grafik Probabilitas")
            st.bar_chart(df_prob.set_index("Sentimen"))

        max_prob = df_prob.loc[df_prob["Probabilitas"].idxmax()]
        st.write(f"**Interpretasi:** Model memprediksi sentimen **{max_prob['Sentimen']}** dengan probabilitas tertinggi sebesar **{max_prob['Probabilitas']:.2f}**.")

        with st.expander("🔎 Lihat Detail Tahapan Pembersihan Teks"):
                st.write("**Teks Asli:**", text)
                st.write("**1. Cleaning:**", hasil["cleaning"])
                st.write("**2. Case Folding:**", hasil["case_folding"])
                st.write("**3. Normalisasi:**", hasil["normalisasi"])
                st.write("**4. Tokenizing:**", hasil["tokenize"])
                st.write("**5. Stopword Removal:**", hasil["stopword_removal"])
                st.write("**6. Stemming (Final):**", hasil["stemming_data"])