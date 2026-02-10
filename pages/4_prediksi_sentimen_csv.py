import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils.loader import load_model
import os

# =============================
# KONFIGURASI HALAMAN
# =============================
st.set_page_config(
    page_title="Prediksi & Evaluasi CSV",
    layout="wide"
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
    '<div class="blue-header">📂 PREDIKSI SENTIMEN CSV</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align:center; color:#7f8c8d;'>
    Halaman ini digunakan untuk melakukan prediksi sentimen otomatis menggunakan model <b>Logistic Regression</b> terhadap data hasil preprocessing.
    </p>
    """,
    unsafe_allow_html=True
)

# =============================
# INFO FILE
# =============================
st.info("""
📌 **Ketentuan File CSV**
- Format file: **.csv**
- Wajib memiliki kolom **stemming_data**
- Kolom lain bersifat opsional
- Evaluasi model hanya dilakukan jika sistem menemukan label aktual
""")

st.markdown("---")

# =============================
# LOAD MODEL
# =============================
try:
    model, tfidf = load_model()
except:
    st.error("⚠️ Model atau TF-IDF gagal dimuat.")
    st.stop()

label_mapping = {
    0: "Negatif",
    1: "Positif"
}

# =============================
# UPLOAD CSV
# =============================
uploaded_file = st.file_uploader(
    "📤 Upload File CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "stemming_data" not in df.columns:
        st.error("❌ Kolom `stemming_data` tidak ditemukan.")
        st.stop()

    st.success(f"✅ File dimuat — {len(df)} data")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    # =============================
    # PREDIKSI SENTIMEN
    # =============================
    with st.spinner("🔍 Melakukan prediksi sentimen..."):
        X = tfidf.transform(df["stemming_data"].astype(str))
        df["label"] = model.predict(X)
        df["sentimen"] = df["label"].map(label_mapping)

    # =============================
    # DISTRIBUSI SENTIMEN
    # =============================
    st.subheader("📊 Distribusi Prediksi Sentimen")

    distribusi = df["sentimen"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.bar(distribusi.index, distribusi.values)
        ax.set_ylabel("Jumlah Data")
        ax.set_title("Distribusi Sentimen")
        st.pyplot(fig)

    with col2:
        st.dataframe(
            distribusi.reset_index().rename(
                columns={"index": "Sentimen", "sentimen": "Jumlah"}
            ),
            use_container_width=True
        )

    st.markdown("---")

    # =============================
    # DETEKSI LABEL AKTUAL OTOMATIS
    # =============================
    label_aktual_col = None
    y_true = None

    excluded_cols = ["stemming_data", "label", "sentimen"]

    for col in df.columns:
        if col in excluded_cols:
            continue

        unique_vals = df[col].dropna().unique()

        if len(unique_vals) == 2:
            # numerik 0/1
            if set(unique_vals).issubset({0, 1}):
                label_aktual_col = col
                y_true = df[col]
                break

            # teks positif / negatif
            lower_vals = set(str(v).lower() for v in unique_vals)
            if lower_vals.issubset({"positif", "negatif"}):
                label_aktual_col = col
                y_true = df[col].str.lower().map({
                    "negatif": 0,
                    "positif": 1
                })
                break

    # =============================
    # EVALUASI MODEL (JIKA ADA LABEL)
    # =============================
    if label_aktual_col is not None:

        st.subheader("🧪 Evaluasi Model")

        st.info(f"""
📌 Evaluasi dilakukan menggunakan kolom **`{label_aktual_col}`** sebagai label aktual.
Hasil evaluasi menunjukkan performa model terhadap data yang telah berlabel.
""")

        cm = confusion_matrix(y_true, df["label"])

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negatif", "Positif"],
            yticklabels=["Negatif", "Positif"],
            ax=ax_cm
        )
        ax_cm.set_xlabel("Prediksi")
        ax_cm.set_ylabel("Label Aktual")
        st.pyplot(fig_cm)

        report = classification_report(
            y_true,
            df["label"],
            target_names=["Negatif", "Positif"],
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    else:
        st.info("ℹ️ Evaluasi model tidak ditampilkan karena sistem tidak menemukan label aktual pada file.")

    st.markdown("---")

    # =============================
    # KESIMPULAN PREDIKSI
    # =============================
    st.subheader("📝 Kesimpulan Hasil Prediksi")

    total = len(df)
    pos = distribusi.get("Positif", 0)
    neg = distribusi.get("Negatif", 0)

    st.markdown(f"""
Berdasarkan hasil prediksi menggunakan model **Logistic Regression**, dari total **{total} data ulasan** yang dianalisis diperoleh hasil sebagai berikut:

- **{pos} data ({(pos/total)*100:.2f}%)** terklasifikasi sebagai **sentimen positif**
- **{neg} data ({(neg/total)*100:.2f}%)** terklasifikasi sebagai **sentimen negatif**

Hasil ini menunjukkan bahwa sentimen yang paling dominan adalah **sentimen {"positif" if pos > neg else "negatif"}**.

Perlu diperhatikan bahwa hasil ini merupakan **prediksi otomatis oleh model**, dan evaluasi performa model hanya ditampilkan apabila data memiliki label aktual.
""")

    # =============================
    # DOWNLOAD CSV
    # =============================
    st.subheader("⬇️ Unduh Hasil Prediksi")

    csv_result = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV Hasil Prediksi",
        csv_result,
        file_name="hasil_prediksi_sentimen.csv",
        mime="text/csv"
    )
