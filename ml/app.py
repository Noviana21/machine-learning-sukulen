import streamlit as st
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# =========================================================
# KUSTOMISASI TEMA WARNA (HIJAU TOSKA & HITAM) VIA CSS
# =========================================================
st.set_page_config(page_title="Klasifikasi Sukulen", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    h1 { color: #20B2AA !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; }
    .stFileUploader { border: 2px dashed #20B2AA !important; background-color: #1A1C23 !important; border-radius: 10px; padding: 10px; }
    .stFileUploader label { color: #20B2AA !important; font-weight: bold; }
    div.stButton > button:first-child {
        background-color: #008080 !important; color: white !important; border-radius: 8px !important;
        border: none !important; padding: 10px 24px !important; font-size: 16px !important;
        font-weight: bold !important; transition: all 0.3s ease; width: 100%;
    }
    div.stButton > button:first-child:hover { background-color: #20B2AA !important; box-shadow: 0px 0px 12px #20B2AA; transform: scale(1.01); }
    </style>
""", unsafe_allow_html=True)

# 1. Judul dan Deskripsi Web
st.title("🌿 Klasifikasi Morfologi Daun Sukulen")
st.write("Silakan seret dan lepas (*drag and drop*) atau pilih satu gambar daun sukulen untuk memprediksi karakteristik morfologinya menggunakan arsitektur VGG16.")

# 2. Memuat Model
@st.cache_resource
def load_vgg_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model_vgg16_terbaik.keras')
    return load_model(model_path)

try:
    model = load_vgg_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file model berada di folder yang sama. Detail: {e}")

# 3. Fitur Upload Gambar
uploaded_file = st.file_uploader(
    "Unggah File Gambar Sukulen (Maksimal 1 Gambar)", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Mengonversi gambar ke format RGB untuk mencegah error pada file PNG transparan
    img = Image.open(uploaded_file).convert('RGB')
    
    st.image(img, caption='Gambar Sukulen yang Siap Dievaluasi', use_container_width=True)
    st.info("💡 *Tips: Anda dapat mengganti gambar kapan saja dengan mengklik ikon silang (X) pada kotak upload di atas atau langsung menyeret gambar baru.*")
    
    # =========================================================
    # PREPROCESSING YANG BENAR (MENYAMAKAN DENGAN COLAB)
    # =========================================================
    img_resized = img.resize((224, 224)) 
    img_array = image.img_to_array(img_resized) 
    
    # BARIS PEMBAGIAN 255 DIHAPUS DARI SINI
    # Agar data yang masuk berformat 0-255 persis seperti tf.keras.utils.image_dataset_from_directory
    
    img_array = np.expand_dims(img_array, axis=0) 

    # 4. Tombol Prediksi
    if st.button("Mulai Prediksi Bentuk Daun"):
        with st.spinner("Menjalankan komputasi ekstraksi fitur dan klasifikasi gambar..."):
            time.sleep(0.5) 
            
            # Eksekusi Prediksi
            pred = model.predict(img_array)
            prob = pred[0][0]  
            
            # Kalkulasi Probabilitas (Asumsi 0 = Bulat, 1 = Runcing)
            if prob > 0.5:
                kelas = "Daun Runcing"
                akurasi = prob * 100
            else:
                kelas = "Daun Bulat"
                akurasi = (1 - prob) * 100
        
        # =========================================================
        # LOGIKA DETEKSI < 60% UNTUK GAMBAR BUKAN DAUN
        # =========================================================
        if akurasi < 60.0:
            st.warning(f"⚠️ **Peringatan Batas Ambang Keputusan!** Tingkat keyakinan model berada di bawah batas minimum (hanya {akurasi:.2f}%).")
            st.error("Gambar terdeteksi memiliki probabilitas anomali yang tinggi. Objek kemungkinan besar **BUKAN bagian dari morfologi daun sukulen** yang valid, atau Anda harus menggunakan gambar dengan sudut pandang yang lebih jelas.")
        else:
            st.success(f"🎉 **Analisis Selesai!** Model mengidentifikasi objek sebagai: **{kelas}**")
            st.metric(label="Tingkat Keyakinan Klasifikasi (Confidence Score)", value=f"{akurasi:.2f}%")