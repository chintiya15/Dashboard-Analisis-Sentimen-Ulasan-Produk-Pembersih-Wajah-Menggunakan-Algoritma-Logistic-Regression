import streamlit as st
import pandas as pd
import os

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Cek Ulasan Produk Berdasarkan Jenis Kulit",
    layout="wide"
)

# =============================
# CUSTOM CSS
# =============================
def load_css():
    css_path = os.path.join("assets", "style.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =============================
# HEADER
# =============================
st.markdown('<div class="blue-header">🔍 CEK ULASAN BERDASARKAN JENIS KULIT</div>', unsafe_allow_html=True)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("hasil_preprocessing_data.csv")

df = load_data()

label_mapping = {0: "Negatif", 1: "Netral", 2: "Positif"}
if "sentimen" not in df.columns:
    df["sentimen"] = df["label"].map(label_mapping)

# =============================
# KEYWORD JENIS KULIT (SMART FILTER)
# =============================
keyword_kulit = {
    "Berminyak": "berminyak|oily|minyak",
    "Kering": "kering|dry",
    "Sensitif": "sensitif|sensitive",
    "Normal": "normal",
    "Kombinasi": "kombinasi|combination"
}

# =============================
# INPUT USER
# =============================
col_in1, col_in2 = st.columns(2)

with col_in1:
    jenis_kulit = st.selectbox(
        "Pilih Jenis Kulit:",
        list(keyword_kulit.keys())
    )

with col_in2:
    produk = st.text_input(
        "Masukkan nama produk:",
        placeholder="Contoh: Garnier Hijau"
    )

# =============================
# PROSES ANALISIS
# =============================
if st.button("📊 Cek Ulasan"):

    if produk.strip() == "":
        st.warning("⚠️ Silakan masukkan nama produk.")
    else:
        data_produk = df[
            (df["full_text"].str.contains(produk, case=False, na=False)) &
            (df["full_text"].str.contains(keyword_kulit[jenis_kulit], case=False, na=False))
        ]

        if data_produk.empty:
            st.error("❌ Tidak ditemukan ulasan sesuai produk dan jenis kulit.")
        else:
            st.success(
                f"✅ Ditemukan **{len(data_produk)} ulasan** untuk "
                f"produk **{produk}** pada kulit **{jenis_kulit}**"
            )

            distribusi = (
                data_produk["sentimen"]
                .value_counts()
                .reindex(["Negatif", "Netral", "Positif"], fill_value=0)
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("😠 Negatif", distribusi["Negatif"])
            m2.metric("😐 Netral", distribusi["Netral"])
            m3.metric("😊 Positif", distribusi["Positif"])

            sentimen_dominan = distribusi.idxmax()
            persentase = (distribusi.max() / distribusi.sum()) * 100

            st.info(
                f"Produk **{produk}** didominasi sentimen "
                f"**{sentimen_dominan}** ({persentase:.2f}%) "
                f"pada konteks kulit **{jenis_kulit}**."
            )

            st.subheader("📝 Contoh Ulasan Pengguna")
            st.dataframe(
                data_produk[["full_text", "sentimen"]].head(10),
                use_container_width=True
            )
