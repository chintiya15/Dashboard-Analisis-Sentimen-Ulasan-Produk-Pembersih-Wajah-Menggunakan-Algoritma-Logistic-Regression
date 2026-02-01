import streamlit as st

st.set_page_config(
    page_title="Analisis Sentimen Pembersih Wajah",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AUTO PINDAH KE DASHBOARD
st.switch_page("pages/1_dashboard.py")
