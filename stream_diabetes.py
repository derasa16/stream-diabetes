import tensorflow as tf
import streamlit as st
import numpy as np
import os
import pickle

# Load TensorFlow model
model_path = 'C:/Users/ACER/Downloads/Big project ann/diabetes_model.h5'
scaler_path = 'C:/Users/ACER/Downloads/Big project ann/robust_scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    try:
        diabetes_model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        st.write("Model and scaler loaded successfully")
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
else:
    st.error("Model or scaler file not found")

# Judul web
st.title('Prediksi Pasien Terkena Diabetes')

# Membagi kolom
col1,col2 = st.columns(2)
with col1 :
    Age = st.text_input('Input Usia Pasien (Tahun)')
with col2 :
    Blood_Glucose_Level = st.text_input('Input Kadar Gula Darah Pasien (mg/dl)')
with col1 :
    Diastolic_Blood_Pressure = st.text_input('Input Tekanan Darah Diastolik Pasien (mmHg)')
with col2 :
    Systolic_Blood_Pressure = st.text_input('Input Tekanan Darah Sistolik Pasien (mmHg)')
with col1 :
    Heart_Rate = st.text_input('Input Detak Jantung Pasien (bpm)')
with col2 :
    Body_Temperature = st.text_input('Input Suhu Tubuh Pasien (Fahrenheit)')
with col1 :
    SPO2 = st.text_input('Input SP02 Pasien (%)')
with col2 :
    Sweating = st.text_input('Apakah Pasien Mengalami Kondisi Tubuh Berkeringat? (0=Tidak, 1=Ya)')
with col1 :
    Shivering = st.text_input('Apakah Pasien Mengalami Kondisi Tubuh Gemetaran? (0=Tidak, 1=Ya)')

# Code untuk prediksi
diab_diagnosis = ''

# Membuat tombol untuk prediksi
if st.button('Test Prediksi Diabetes'):
    try:
        # Pastikan input dikonversi ke tipe data yang sesuai
        input_data = np.array([[int(Age), int(Blood_Glucose_Level), int(Diastolic_Blood_Pressure),
                                int(Systolic_Blood_Pressure), int(Heart_Rate), float(Body_Temperature),
                                int(SPO2), int(Sweating), int(Shivering)]])
        
        # Normalisasi data input menggunakan scaler yang sudah dilatih
        input_data = scaler.transform(input_data)
        
        # Lakukan prediksi
        diab_prediction = diabetes_model.predict(input_data)
        
        # Jika model mengembalikan nilai biner, gunakan argmax
        if np.argmax(diab_prediction) == 1:
            diab_diagnosis = 'Pasien Terkena Diabetes'
        else:
            diab_diagnosis = 'Pasien Tidak Terkena Diabetes'

        st.success(diab_diagnosis)
    except ValueError as e:
        st.error(f'Error dalam konversi input: {e}')
    except Exception as e:
        st.error(f'Error: {e}')
