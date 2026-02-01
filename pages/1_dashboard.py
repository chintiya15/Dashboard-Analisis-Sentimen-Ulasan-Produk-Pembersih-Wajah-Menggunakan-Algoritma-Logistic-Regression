import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    layout="wide",
    initial_sidebar_state="expanded"
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
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("hasil_preprocessing_data.csv")
    except:
        return pd.DataFrame({'full_text': [], 'label': [], 'steming_data': []})

df = load_data()
label_mapping = {0: "Negatif", 1: "Netral", 2: "Positif"}

# =============================
# HEADER
# =============================
st.markdown('<div class="blue-header">📊 DASHBOARD ANALISIS SENTIMEN</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: #7f8c8d; margin-bottom:30px;'>Ulasan Produk Pembersih Wajah</p>", unsafe_allow_html=True)
st.info("Dashboard ini menyajikan ringkasan hasil analisis sentimen menggunakan algoritma **Logistic Regression**.")

# =============================
# PROSES DATA
# =============================
if not df.empty:
    if "sentimen" not in df.columns:
        df["sentimen"] = df["label"].map(label_mapping)

    jumlah_sentimen = df["sentimen"].value_counts()
    akurasi_model = 0.72

    # =============================
    # METRIC CARDS DENGAN BACKGROUND
    # =============================
    col1, col2, col3, col4, col5 = st.columns(5)

    def metric_card(judul, nilai):
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{judul}</div>
            <div class="metric-value">{nilai}</div>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        metric_card("📄 Total Data", len(df))
    with col2:
        metric_card("😊 Positif", jumlah_sentimen.get("Positif", 0))
    with col3:
        metric_card("😐 Netral", jumlah_sentimen.get("Netral", 0))
    with col4:
        metric_card("😠 Negatif", jumlah_sentimen.get("Negatif", 0))
    with col5:
        metric_card("🎯 Akurasi", f"{akurasi_model*100:.2f}%")

    st.markdown("---")

    # =============================
    # DISTRIBUSI SENTIMEN
    # =============================
    st.subheader("📊 Distribusi Sentimen")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.pie(
        jumlah_sentimen.values,
        labels=jumlah_sentimen.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=['#FF7F0E', '#2CA02C', '#1F77B4'],
        textprops={'color':'black'}
    )
    fig.patch.set_facecolor('white')
    st.pyplot(fig)

    st.markdown("---")

    # =============================
    # WORDCLOUD & FREKUENSI
    # =============================
    st.subheader("☁️ WordCloud & Frekuensi Kata")

    sentimen_pilihan = st.selectbox("Pilih Sentimen", ["Positif", "Netral", "Negatif"])
    df_filtered = df[df["sentimen"] == sentimen_pilihan]
    teks = " ".join(df_filtered["steming_data"].astype(str))

    if teks.strip():
        col_wc, col_bar = st.columns(2)

        with col_wc:
            wc = WordCloud(width=800, height=500, background_color="white", colormap="Blues").generate(teks)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc)
            ax_wc.axis("off")
            st.pyplot(fig_wc)

        with col_bar:
            kata = Counter(teks.split()).most_common(10)  # TOP 10 SAJA
            words, counts = zip(*kata)

            fig_bar, ax_bar = plt.subplots(figsize=(12,6))

            bars = ax_bar.bar(
                words,
                counts,
                color=[
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                    "#bcbd22", "#17becf"
                ]
            )

            # Judul & label
            ax_bar.set_title("Frekuensi Kata", fontsize=18, fontweight="bold")
            ax_bar.set_xlabel("Kata-Kata Sering Muncul", fontsize=12)
            ax_bar.set_ylabel("Jumlah Kata", fontsize=12)

            # Rotasi agar tidak tabrakan
            plt.xticks(rotation=45, ha="right")

            # Angka di atas bar
            for bar in bars:
                height = bar.get_height()
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=10
                )

            fig_bar.patch.set_facecolor("white")
            ax_bar.set_facecolor("white")

            st.pyplot(fig_bar)



    st.markdown("---")
    st.subheader("📋 Contoh Data")
    # st.dataframe(df[["full_text","sentimen"]].head(10), use_container_width=True)
    # Ambil masing-masing 3 data secara acak per sentimen
    sample_pos = df[df["sentimen"] == "Positif"].sample(n=min(3, len(df[df["sentimen"]=="Positif"])), random_state=None)
    sample_neg = df[df["sentimen"] == "Negatif"].sample(n=min(3, len(df[df["sentimen"]=="Negatif"])), random_state=None)
    sample_net = df[df["sentimen"] == "Netral"].sample(n=min(3, len(df[df["sentimen"]=="Netral"])), random_state=None)

    # Gabungkan lalu acak ulang urutannya
    sample_data = pd.concat([sample_pos, sample_neg, sample_net]).sample(frac=1)

    # Tampilkan
    st.dataframe(sample_data[["full_text", "sentimen"]], use_container_width=True)


else:
    st.error("Data tidak ditemukan.")
