import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import os

# Import loader dari folder utils
from utils.loader import load_model

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
# 3. LOAD DATA & MODEL
# ==============================================================================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("hasil_preprocessing_data.csv")
    except:
        return pd.DataFrame()

df = load_data()
model, tfidf = load_model()

# ==============================================================================
# 4. DATA PENGUJIAN HASIL SENTIMEN
# ==============================================================================
# Confusion Matrix Data: [[TN, FP], [FN, TP]]
cm_data = [[6, 33], [2, 194]]
tn, fp, fn, tp = 6, 33, 2, 194

# Data Classification Report Manual
report_dict = {
    "Negatif": {"precision": 1.00, "recall": 0.08, "f1-score": 0.14, "support": 39},
    "Positif": {"precision": 0.85, "recall": 1.00, "f1-score": 0.92, "support": 197},
    "accuracy": 0.85,
    "macro avg": {"precision": 0.92, "recall": 0.54, "f1-score": 0.53, "support": 236},
    "weighted avg": {"precision": 0.87, "recall": 0.85, "f1-score": 0.79, "support": 236}
}

akurasi_asli = report_dict["accuracy"]

if "sentimen" not in df.columns:
    df["sentimen"] = df["validasi_label"].map({0: "Negatif", 1: "Positif"})

jumlah_sentimen = df["sentimen"].value_counts()

# ==============================================================================
# 5. HEADER DASHBOARD
# ==============================================================================
st.markdown('<div class="blue-header">📊 DASHBOARD ANALISIS SENTIMEN</div>', unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:#7f8c8d; margin-bottom:25px;'>"
    "Analisis Sentimen Ulasan Produk Pembersih Wajah pada Platform Twitter/X"
    "</p>",
    unsafe_allow_html=True
)

st.info(f"Dashboard ini menampilkan hasil analisis menggunakan **Logistic Regression**. Akurasi Pengujian: **{akurasi_asli*100:.2f}%**")

st.markdown("---")

# ==============================================================================
# 6. METRIC CARD
# ==============================================================================
st.subheader("📌 Ringkasan Data Sentimen")
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
    metric_card("🎯 Akurasi Uji", f"{akurasi_asli*100:.2f}%")

st.markdown("---")

# ==============================================================================
# 7. DISTRIBUSI SENTIMEN
# ==============================================================================
st.subheader("📊 Distribusi Sentimen Tweet")
fig, ax = plt.subplots(figsize=(6, 4))
ax.pie(jumlah_sentimen.values, labels=jumlah_sentimen.index, autopct="%1.1f%%", startangle=90)
st.pyplot(fig)

st.markdown("---")

# ==============================================================================
# 8. WORDCLOUD & FREKUENSI KATA
# ==============================================================================
st.subheader("☁️ WordCloud dan Kata yang Sering Muncul")
sentimen_pilihan = st.selectbox("Pilih Sentimen:", ["Positif", "Negatif"])

df_filtered = df[df["sentimen"] == sentimen_pilihan]
teks = " ".join(df_filtered["stemming_data"].astype(str))

if teks.strip() != "":
    col_wc, col_bar = st.columns(2)
    with col_wc:
        wc = WordCloud(width=800, height=500, background_color="white").generate(teks)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc)
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    with col_bar:
        kata = Counter(teks.split()).most_common(10)
        words, counts = zip(*kata)
        fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
        ax_bar.bar(words, counts)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_bar)

st.markdown("---")

# ==============================================================================
# 9. CONTOH TWEET (FIXED WRAP TEXT)
# ==============================================================================
st.subheader("📋 Contoh Tweet Ulasan Produk Pembersih Wajah")

sample_pos = df[df["sentimen"] == "Positif"].sample(
    n=min(5, len(df[df["sentimen"] == "Positif"])),
    random_state=42
)

sample_neg = df[df["sentimen"] == "Negatif"].sample(
    n=min(5, len(df[df["sentimen"] == "Negatif"])),
    random_state=42
)

sample_data = pd.concat([sample_pos, sample_neg]).sample(frac=1)

st.dataframe(
    sample_data[["full_text", "sentimen"]],
    use_container_width=True
)

# ==============================================================================
# 10. EVALUASI MODEL (CONFUSION MATRIX & REPORT)
# ==============================================================================
st.subheader("🎯 Evaluasi Performa Model")

col_cm, col_desc = st.columns([1.2, 1])

with col_cm:
    z = cm_data
    x = ['Prediksi Negatif', 'Prediksi Positif']
    y = ['Aktual Negatif', 'Aktual Positif']

    fig_cm = px.imshow(
        z, x=x, y=y, 
        text_auto=True, 
        color_continuous_scale='Blues',
        labels=dict(x="Hasil Prediksi", y="Data Aktual", color="Jumlah")
    )
    fig_cm.update_layout(title_text='Confusion Matrix')
    st.plotly_chart(fig_cm, use_container_width=True)

with col_desc:
    st.markdown(f"""
    **Keterangan:**
    
    * ✅ **TP (True Positive) : {tp}** Sampel positif yang berhasil diprediksi dengan benar sebagai positif.
      
    * ✅ **TN (True Negative) : {tn}** Sampel negatif yang berhasil diprediksi dengan benar sebagai negatif.
      
    * ❌ **FP (False Positive) : {fp}** Sampel negatif yang salah diprediksi sebagai positif.
      
    * ❌ **FN (False Negative) : {fn}** Sampel positif yang salah diprediksi sebagai negatif.
    """)

# Classification Report Table
st.write("**Classification Report**")
report_df = pd.DataFrame({
    "precision": [1.00, 0.85, None, 0.92, 0.87],
    "recall": [0.08, 1.00, None, 0.54, 0.85],
    "f1-score": [0.14, 0.92, 0.85, 0.53, 0.79],
    "support": [39, 197, 236, 236, 236]
}, index=["Negatif", "Positif", "accuracy", "macro avg", "weighted avg"])

st.table(report_df.fillna("-").style.format(precision=2))

# Penjelasan Berdasarkan Hasil
st.markdown(f"""
### 📊 Penjelasan Metrik Evaluasi

1. **Accuracy (Akurasi)**: Merupakan ukuran tingkat kedekatan antara hasil prediksi dengan nilai sebenarnya. Berdasarkan hasil, sistem memiliki akurasi sebesar **{report_dict['accuracy']:.2f}**, yang berarti **85%** data uji berhasil diklasifikasikan dengan benar.

2. **Precision (Presisi)**: Merupakan ukuran yang menunjukkan jumlah dokumen relevan dari keseluruhan dokumen yang berhasil ditemukan oleh sistem. Berdasarkan hasil precision, kelas **Negatif memiliki nilai 1.00 (100%)**, artinya setiap kali sistem memprediksi negatif, prediksi tersebut selalu benar. Sedangkan kelas **Positif memiliki nilai 0.85 (85%)**.

3. **Recall**: Berfungsi sebagai alat ukur untuk menilai tingkat efektivitas sistem dalam menemukan kembali informasi. Berdasarkan hasil, kelas **Positif memiliki recall sempurna 1.00 (100%)**, namun kelas **Negatif hanya memiliki recall 0.08 (8%)**, yang menunjukkan sistem masih kesulitan mengidentifikasi ulasan negatif secara keseluruhan.

4. **F1-Score**: Merupakan metrik evaluasi yang menggabungkan recall dan precision. Nilai F1-Score sebesar **0.92** pada kelas positif menunjukkan keseimbangan performa yang sangat baik untuk ulasan positif.
""")

# ==============================================================================
# 11. KESIMPULAN
# ==============================================================================
st.markdown("---")
st.subheader("📝 Kesimpulan")
if jumlah_sentimen.get("Positif", 0) > jumlah_sentimen.get("Negatif", 0):
    st.success("Mayoritas ulasan menunjukkan sentimen **positif**, yang menandakan produk diterima dengan baik oleh pengguna.")
else:
    st.warning("Mayoritas ulasan menunjukkan sentimen **negatif**, perlu dilakukan evaluasi kualitas produk.")