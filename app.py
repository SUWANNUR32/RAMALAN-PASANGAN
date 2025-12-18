import streamlit as st
import datetime
import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="AI Weton Predictor", page_icon="🔮")

# 1. SETUP MODEL AI (Caching agar tidak load berulang kali)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_model()

def get_weton(tanggal_lahir):
    hari_map = {"Sunday": 5, "Monday": 4, "Tuesday": 3, "Wednesday": 7, "Thursday": 8, "Friday": 6, "Saturday": 9}
    pasaran_list = ["Kliwon", "Legi", "Pahing", "Pon", "Wage"]
    pasaran_map = {"Kliwon": 8, "Legi": 5, "Pahing": 9, "Pon": 7, "Wage": 4}
    
    # Konversi dari object date streamlit ke datetime
    dt = datetime.datetime.combine(tanggal_lahir, datetime.time())
    hari_eng = dt.strftime("%A")
    base_date = datetime.datetime(1900, 1, 1) 
    delta_days = (dt - base_date).days
    pasaran_name = pasaran_list[(delta_days + 2) % 5] 
    
    neptu_total = hari_map[hari_eng] + pasaran_map[pasaran_name]
    return neptu_total, f"{hari_eng} {pasaran_name}"

def interpretasi_tibo(sisa):
    tibo_data = {
        1: ("PEGAT", "Risiko perpisahan, masalah ekonomi, atau kekuasaan."),
        2: ("RATU", "Sangat harmonis, disegani tetangga, dan sudah jodohnya."),
        3: ("JODOH", "Sangat cocok, rukun, dan bisa menerima kekurangan masing-masing."),
        4: ("TOPO", "Susah di awal pernikahan, namun akan sukses di masa depan."),
        5: ("TINARI", "Murah rezeki, sering beruntung, dan mudah mencari nafkah."),
        6: ("PADU", "Sering bertengkar namun tidak sampai bercerai (tetap rukun)."),
        7: ("SUJANAN", "Rentan isu perselingkuhan atau pertengkaran hebat."),
        0: ("PESTHI", "Kehidupan damai, rukun, dan tentram hingga tua.")
    }
    return tibo_data.get(sisa, ("Tidak Diketahui", "-"))

def draw_gauge(score, label_tibo):
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': 'polar'})
    colors = ['#ff4b4b', '#f9d423', '#00d084']
    values = [0, 50, 75, 100]
    for i in range(len(colors)):
        ax.barh(1, np.radians(values[i+1]-values[i]), left=np.radians(values[i]), color=colors[i], height=0.5)
    pos = np.radians(score)
    ax.annotate('', xy=(pos, 0), xytext=(pos, 1.2), arrowprops=dict(arrowstyle='wedge', color='black', lw=3))
    ax.set_theta_zero_location("N", offset=90)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([0, 50, 75, 100], labels=['Bahaya', 'Cukup', 'Baik', 'Sangat Baik'])
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    plt.title(f"Relationship Score: {score:.1f}%\nStatus: {label_tibo}", va='bottom', fontsize=12, fontweight='bold')
    return fig

# --- ANTARMUKA STREAMLIT ---
st.title("🔮 AI Hybrid Weton Predictor")
st.write("Analisis kecocokan pasangan berdasarkan Primbon Jawa dan AI Sentiment Analysis.")

col1, col2 = st.columns(2)

with col1:
    tgl_pria = st.date_input("Tanggal Lahir Pria", value=datetime.date(1995, 1, 1))
with col2:
    tgl_wanita = st.date_input("Tanggal Lahir Wanita", value=datetime.date(1995, 1, 1))

curhatan = st.text_area("Ceritakan kondisi hubungan Anda saat ini (dalam Bahasa Inggris):", 
                        placeholder="Example: I feel very happy and we always support each other...")

if st.button("Analisis Hubungan"):
    if curhatan:
        # 1. Hitung Weton
        n_pria, w_pria = get_weton(tgl_pria)
        n_wanita, w_wanita = get_weton(tgl_wanita)
        total_neptu = n_pria + n_wanita
        sisa_tibo = total_neptu % 8
        nama_tibo, deskripsi_tibo = interpretasi_tibo(sisa_tibo)

        # 2. Analisis Sentimen
        res = classifier(curhatan)[0]
        sentiment_score = res['score'] * 100 if res['label'] == 'POSITIVE' else (1 - res['score']) * 100

        # 3. Skor Final
        weton_weight_map = {1:30, 2:95, 3:90, 4:70, 5:85, 6:60, 7:40, 0:90}
        final_score = (weton_weight_map.get(sisa_tibo, 50) + sentiment_score) / 2

        # --- DISPLAY HASIL ---
        st.divider()
        st.subheader("Hasil Analisis")
        
        c1, c2 = st.columns(2)
        c1.metric("Weton Pria", w_pria, f"Neptu {n_pria}")
        c2.metric("Weton Wanita", w_wanita, f"Neptu {n_wanita}")

        st.info(f"**Tibo {nama_tibo}**: {deskripsi_tibo}")

        # Penjelasan Psikologis
        st.write("### Skor Psikologis AI")
        if sentiment_score >= 80:
            st.success(f"Skor: {sentiment_score:.1f}% - AI mendeteksi rasa percaya dan kebahagiaan yang kuat.")
        elif sentiment_score >= 50:
            st.warning(f"Skor: {sentiment_score:.1f}% - Hubungan cenderung aman, namun ada ruang untuk lebih terbuka.")
        else:
            st.error(f"Skor: {sentiment_score:.1f}% - AI mendeteksi adanya kecemasan atau tekanan emosional.")

        # Grafik
        st.pyplot(draw_gauge(final_score, nama_tibo))
    else:
        st.error("Mohon isi curhatan terlebih dahulu untuk analisis AI.")
