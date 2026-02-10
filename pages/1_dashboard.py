import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
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
# 3. LOAD DATASET
# ==============================================================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("hasil_preprocessing_data.csv")
    except:
        return pd.DataFrame()

df = load_data()

# ==============================================================================
# 4. LABEL MAPPING
# ==============================================================================
label_mapping = {
    0: "Negatif",
    1: "Positif"
}

# ==============================================================================
# 5. HEADER DASHBOARD
# ==============================================================================
st.markdown(
    '<div class="blue-header">📊 DASHBOARD ANALISIS SENTIMEN</div>',
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#7f8c8d; margin-bottom:25px;'>"
    "Analisis Sentimen Ulasan Produk Pembersih Wajah pada Platform Twitter/X"
    "</p>",
    unsafe_allow_html=True
)

st.info(
    "Dashboard ini menampilkan **hasil akhir klasifikasi sentimen** menggunakan "
    "algoritma **Logistic Regression dengan SMOTE** sebagai model terbaik."
)

st.markdown("---")

# ==============================================================================
# 6. CEK DATASET
# ==============================================================================
if df.empty:
    st.error("⚠️ Dataset tidak ditemukan. Pastikan file hasil_preprocessing_data.csv tersedia.")
    st.stop()

# ==============================================================================
# 7. VALIDASI KOLOM WAJIB
# ==============================================================================
kolom_wajib = ["full_text", "validasi_label", "stemming_data"]

for col in kolom_wajib:
    if col not in df.columns:
        st.error(f"⚠️ Kolom **{col}** tidak ditemukan dalam dataset.")
        st.stop()

# ==============================================================================
# 8. TAMBAHKAN KOLOM SENTIMEN
# ==============================================================================
if "sentimen" not in df.columns:
    df["sentimen"] = df["validasi_label"].map(label_mapping)

jumlah_sentimen = df["sentimen"].value_counts()

# ==============================================================================
# 9. METRIC CARD (MODEL FINAL)
# ==============================================================================
akurasi_sebelum_smote = 0.8559
akurasi_sesudah_smote = 0.8771  # MODEL FINAL

st.subheader("📌 Ringkasan Data Sentimen (Model Final)")

col1, col2, col3, col4 = st.columns(4)

def metric_card(judul, nilai):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{judul}</div>
            <div class="metric-value">{nilai}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col1:
    metric_card("📄 Total Tweet", len(df))

with col2:
    metric_card("😊 Positif", jumlah_sentimen.get("Positif", 0))

with col3:
    metric_card("😠 Negatif", jumlah_sentimen.get("Negatif", 0))

with col4:
    metric_card("🎯 Akurasi Model (SMOTE)", f"{akurasi_sesudah_smote*100:.2f}%")

st.markdown("---")

# ==============================================================================
# 10. PERBANDINGAN SEBELUM & SESUDAH SMOTE
# ==============================================================================
st.subheader("📊 Perbandingan Model Sebelum dan Sesudah SMOTE")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### ❌ Sebelum SMOTE")
    st.metric("Akurasi Model", f"{akurasi_sebelum_smote*100:.2f}%")
    st.write("Distribusi kelas tidak seimbang, sehingga model cenderung bias.")

with col_b:
    st.markdown("### ✅ Sesudah SMOTE (Model Final)")
    st.metric("Akurasi Model", f"{akurasi_sesudah_smote*100:.2f}%")
    st.write("Distribusi kelas lebih seimbang dan performa model meningkat.")

st.markdown("---")

# ==============================================================================
# 11. DISTRIBUSI SENTIMEN
# ==============================================================================
st.subheader("📊 Distribusi Sentimen Tweet")

fig, ax = plt.subplots(figsize=(6, 4))
ax.pie(
    jumlah_sentimen.values,
    labels=jumlah_sentimen.index,
    autopct="%1.1f%%",
    startangle=90
)
st.pyplot(fig)

st.markdown("---")

# ==============================================================================
# 12. WORDCLOUD & FREKUENSI KATA
# ==============================================================================
st.subheader("☁️ WordCloud dan Kata yang Sering Muncul")

sentimen_pilihan = st.selectbox(
    "Pilih Sentimen untuk Ditampilkan:",
    ["Positif", "Negatif"]
)

df_filtered = df[df["sentimen"] == sentimen_pilihan]
teks = " ".join(df_filtered["stemming_data"].astype(str))

if teks.strip() == "":
    st.warning("Tidak ada teks yang dapat divisualisasikan.")
else:
    col_wc, col_bar = st.columns(2)

    with col_wc:
        wc = WordCloud(
            width=800,
            height=500,
            background_color="white"
        ).generate(teks)

        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    with col_bar:
        kata = Counter(teks.split()).most_common(10)

        if kata:
            words, counts = zip(*kata)
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            ax_bar.bar(words, counts)
            ax_bar.set_xlabel("Kata")
            ax_bar.set_ylabel("Jumlah")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig_bar)

st.markdown("---")

# ==============================================================================
# 13. CONTOH DATA ULASAN
# ==============================================================================
st.subheader("📋 Contoh Tweet Berdasarkan Sentimen")

sample_pos = df[df["sentimen"] == "Positif"].sample(
    n=min(3, len(df[df["sentimen"] == "Positif"])),
    random_state=42
)

sample_neg = df[df["sentimen"] == "Negatif"].sample(
    n=min(3, len(df[df["sentimen"] == "Negatif"])),
    random_state=42
)

sample_data = pd.concat([sample_pos, sample_neg]).sample(frac=1)

st.dataframe(
    sample_data[["full_text", "sentimen"]],
    use_container_width=True
)

# ==============================================================================
# 14. KESIMPULAN
# ==============================================================================
st.markdown("---")
st.subheader("📝 Kesimpulan")

if jumlah_sentimen.get("Positif", 0) > jumlah_sentimen.get("Negatif", 0):
    st.success(
        "Mayoritas ulasan menunjukkan sentimen **positif**, yang menandakan "
        "produk pembersih wajah diterima dengan baik oleh pengguna."
    )
else:
    st.warning(
        "Mayoritas ulasan menunjukkan sentimen **negatif**, sehingga "
        "produsen perlu melakukan evaluasi lebih lanjut."
    )
