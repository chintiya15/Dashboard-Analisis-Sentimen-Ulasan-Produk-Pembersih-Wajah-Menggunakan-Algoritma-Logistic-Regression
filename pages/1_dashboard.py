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
cm_data = [[3, 17], [0, 98]]
tn, fp, fn, tp = 3, 17, 0, 98

# Data Classification Report Manual
report_dict = {
    "Negatif": {"precision": 1.0000, "recall": 0.1500, "f1-score": 0.2609, "support": 20},
    "Positif": {"precision": 0.8522, "recall": 1.0000, "f1-score": 0.9202, "support": 98},
    "accuracy": 0.8559,
    "macro avg": {"precision": 0.9242, "recall": 0.5513, "f1-score": 0.5520, "support": 118},
    "weighted avg": {"precision": 0.8736, "recall": 0.8511, "f1-score": 0.7966, "support": 118}
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
    metric_card("🎯 Akurasi Test", f"{akurasi_asli*100:.2f}%")

st.markdown("---")

# ==============================================================================
# 7. DISTRIBUSI SENTIMEN (VERSI RINGKAS)
# ==============================================================================
st.subheader("📊 Distribusi Sentimen Tweet")

# Menggunakan kolom agar chart tidak melebar ke seluruh layar
col_chart, col_text = st.columns([1, 1.5]) 

with col_chart:
    # Ukuran figure diperkecil menjadi (4, 3) agar lebih proporsional
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Menggunakan warna default Matplotlib seperti sebelumnya
    ax.pie(
        jumlah_sentimen.values, 
        labels=jumlah_sentimen.index, 
        autopct="%1.1f%%", 
        startangle=90
    )
    
    # Menghilangkan margin berlebih di sekitar pie chart
    plt.tight_layout()
    st.pyplot(fig)

with col_text:
    # Memberikan ruang kosong atau teks ringkasan agar tata letak seimbang
    st.write(" ")
    st.write(" ")
    st.markdown(f"""
    **Keterangan:**
    
    Visualisasi ini menunjukkan perbandingan antara ulasan **Positif** dan **Negatif**. 
    Dominasi ulasan saat ini berada pada sentimen **{jumlah_sentimen.idxmax()}** dengan total **{len(df)}** data yang dianalisis.
    """)

st.markdown("---")

# ==============================================================================
# 8. WORDCLOUD & FREKUENSI KATA
# ==============================================================================
st.subheader("☁️ WordCloud dan Kata yang Sering Muncul")

# 1. Definisi Stopwords Tambahan (Agar sinkron dengan WordCloud & Barchart)
from wordcloud import STOPWORDS
custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    'https', 'co', '...', 'amp', 'harga', 'water', 'facial', 
    'wash', 'pakai', 'produk', 'banget', 'saya'
])

sentimen_pilihan = st.selectbox("Pilih Sentimen:", ["Positif", "Negatif"])

# 2. Filter data berdasarkan pilihan
df_filtered = df[df["sentimen"] == sentimen_pilihan]

# 3. Proses Pembersihan Teks (Hapus stopwords dan kata pendek < 3 huruf)
def clean_text_for_viz(text_series):
    all_words = " ".join(text_series.astype(str)).lower().split()
    # Hanya ambil kata yang bukan stopword dan panjangnya > 2 karakter
    cleaned_words = [w for w in all_words if w not in custom_stopwords and len(w) > 2]
    return cleaned_words

list_kata = clean_text_for_viz(df_filtered["stemming_data"])
teks_bersih = " ".join(list_kata)

if teks_bersih.strip() != "":
    # Tentukan parameter visual
    if sentimen_pilihan == "Positif":
        map_warna = "Greens"
        judul_wc = "WordCloud Sentimen Positif"
        judul_bar = "Frekuensi Kata Sentimen Positif"
    else:
        map_warna = "Reds"
        judul_wc = "WordCloud Sentimen Negatif"
        judul_bar = "Frekuensi Kata Sentimen Negatif"

    col_wc, col_bar = st.columns([1, 1.2])
    
    with col_wc:
        # Generate WordCloud dari teks yang sudah bersih
        wc = WordCloud(
            width=800, 
            height=500, 
            background_color="white", 
            colormap=map_warna,
            # Stopwords dikosongkan karena teks sudah kita bersihkan manual di atas
            stopwords=None 
        ).generate(teks_bersih)
        
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis("off")
        ax_wc.set_title(judul_wc, fontsize=15, pad=10)
        st.pyplot(fig_wc)

    with col_bar:
        # Hitung frekuensi dari list_kata yang sama dengan sumber WordCloud
        counts_data = Counter(list_kata).most_common(20)
        words, counts = zip(*counts_data)
        
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
        
        # Gunakan warna warni tab10 agar mirip foto
        colors = plt.cm.tab10(range(len(words)))
        bars = ax_bar.bar(words, counts, color=colors)
        
        # Tambahkan label angka di atas bar (Persis seperti di foto)
        for bar in bars:
            yval = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width()/2, 
                yval + (max(counts) * 0.01),
                int(yval), 
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        ax_bar.set_title(judul_bar, fontsize=14, fontweight='bold')
        ax_bar.set_ylabel("Jumlah Kata", fontweight='bold')
        ax_bar.set_xlabel("Kata-Kata Sering Muncul", fontweight='bold')
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_bar)
else:
    st.warning(f"Tidak ada data teks yang cukup untuk menampilkan visualisasi {sentimen_pilihan}.")

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
    "precision": [1.0000, 0.8522, None, 0.9261, 0.8772],
    "recall": [0.1500, 1.0000, None, 0.5750, 0.8559],
    "f1-score": [0.2609, 0.9202, 0.8559, 0.5905, 0.8084],
    "support": [20, 98, 118, 118, 118]
}, index=["Negatif", "Positif", "accuracy", "macro avg", "weighted avg"])

st.table(report_df.fillna("-").style.format(precision=2))

# Penjelasan Berdasarkan Hasil
# Penjelasan Berdasarkan Hasil
st.markdown(f"""
### 📊 Penjelasan Metrik Evaluasi

1. **Accuracy (Akurasi)**: Merupakan ukuran tingkat kedekatan antara hasil prediksi dengan nilai sebenarnya. Berdasarkan hasil, sistem memiliki akurasi sebesar **{report_dict['accuracy']:.4f}**, yang berarti sekitar **{report_dict['accuracy']*100:.2f}%** data uji berhasil diklasifikasikan dengan benar.

2. **Precision (Presisi)**: Merupakan ukuran yang menunjukkan ketepatan model dalam memprediksi kelas. Berdasarkan hasil, kelas **Negatif memiliki nilai 1.00 (100%)**, artinya setiap kali sistem memprediksi negatif, prediksi tersebut selalu benar. Sedangkan kelas **Positif memiliki nilai {report_dict['Positif']['precision']:.2f} ({report_dict['Positif']['precision']*100:.0f}%)**.

3. **Recall**: Menilai tingkat efektivitas sistem dalam menemukan kembali semua data dari kelas tertentu. Kelas **Positif memiliki recall sempurna 1.00 (100%)**, yang berarti seluruh ulasan positif berhasil diidentifikasi. Namun, kelas **Negatif memiliki recall {report_dict['Negatif']['recall']:.2f} ({report_dict['Negatif']['recall']*100:.1f}%)**, menunjukkan sistem masih kesulitan menangkap sebagian besar ulasan negatif yang ada.

4. **F1-Score**: Merupakan rata-rata harmonik antara precision dan recall. Nilai F1-Score sebesar **{report_dict['Positif']['f1-score']:.2f}** pada kelas positif menunjukkan performa yang sangat stabil, sementara nilai **{report_dict['Negatif']['f1-score']:.2f}** pada kelas negatif menunjukkan perlunya peningkatan performa pada identifikasi sentimen negatif.
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