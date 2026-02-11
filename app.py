import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfigurasi Dasar ---
st.set_page_config(page_title="Analisis & Prediksi Skor Siswa", layout="wide")
sns.set_theme(style="darkgrid") # Tema grafik agar terlihat modern

# --- Fungsi Load Data & Model ---
@st.cache_resource
def load_assets():
    # Load model pipeline
    model = pickle.load(open("model.pkl", "rb"))
    # Load dataset untuk visualisasi (pastikan file exams.csv ada)
    df = pd.read_csv('exams.csv')
    # Rename kolom agar konsisten dengan Colab
    df.rename(columns={
        'race/ethnicity': 'race_ethnicity',
        'parental level of education': 'parental_level_of_education',
        'test preparation course': 'test_preparation_course',
        'math score': 'math_score',
        'reading score': 'reading_score',
        'writing_score': 'writing_score'
    }, inplace=True, errors='ignore')
    return model, df

model, df = load_assets()

# --- Navigasi Sidebar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard Utama", "Exploratory Data Analysis", "Prediksi Skor"])

# --- Halaman 1: Dashboard Utama ---
if page == "Dashboard Utama":
    st.title("🚀 Student Performance Analytics")
    st.markdown("""
    ### Business Understanding
    Aplikasi ini bertujuan untuk mengidentifikasi variabel penting yang memengaruhi skor ujian siswa. 
    Dengan model ini, pihak institusi dapat melakukan intervensi dini bagi siswa yang diprediksi memiliki performa rendah.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", len(df))
    col2.metric("Rata-rata Math Score", round(df['math_score'].mean(), 2))
    col3.metric("Korelasi Tertinggi", "0.95 (Read vs Write)")

    st.subheader("Cuplikan Data")
    st.dataframe(df.head(10), use_container_width=True)

# --- Halaman 2: Exploratory Data Analysis (EDA) ---
elif page == "Exploratory Data Analysis":
    st.title("📊 Visualisasi Data Mendalam")
    
    tab_dist, tab_corr, tab_cat = st.tabs(["Distribusi Skor", "Analisis Korelasi", "Analisis Kategorikal"])

    with tab_dist:
        st.subheader("Distribusi Skor Matematika")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['math_score'], kde=True, color='royalblue', ax=ax)
        plt.title("Penyebaran Nilai Matematika Siswa")
        st.pyplot(fig)
        st.info("Insight: Mayoritas siswa berada di rentang nilai 60-80, menunjukkan distribusi normal.")

    with tab_corr:
        st.subheader("Matriks Korelasi")
        fig, ax = plt.subplots(figsize=(8, 6))
        # Hanya hitung korelasi data numerik
        corr = df.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        st.write("Hubungan linear yang kuat antara skor membaca dan menulis mendukung performa matematika.")

    with tab_cat:
        st.subheader("Perbandingan Skor Berdasarkan Kategori")
        cat_feature = st.selectbox("Pilih Variabel Kategori:", 
                                  ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=cat_feature, y='math_score', data=df, palette='viridis', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

# --- Halaman 3: Prediksi Skor ---
elif page == "Prediksi Skor":
    st.title("🔮 Prediksi Skor Matematika")
    st.write("Masukkan parameter siswa di bawah ini untuk mendapatkan hasil prediksi.")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", df['gender'].unique())
            race = st.selectbox("Race/Ethnicity", df['race_ethnicity'].unique())
            parent = st.selectbox("Parental Education", df['parental_level_of_education'].unique())
            lunch = st.selectbox("Lunch Type", df['lunch'].unique())
        
        with c2:
            prep = st.selectbox("Test Prep Course", df['test_preparation_course'].unique())
            reading = st.slider("Reading Score", 0, 100, 70)
            writing = st.slider("Writing Score", 0, 100, 70)

        if st.button("Hitung Prediksi", use_container_width=True):
            input_df = pd.DataFrame([{
                'gender': gender,
                'race_ethnicity': race,
                'parental_level_of_education': parent,
                'lunch': lunch,
                'test_preparation_course': prep,
                'reading_score': reading,
                'writing_score': writing
            }])
            
            prediction = model.predict(input_df)[0]
            
            st.divider()
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown(f"### Prediksi Skor: **{round(prediction, 2)}**")
                status = "LULUS" if prediction >= 60 else "REMIDIAL"
                st.markdown(f"Status: <span style='color:{'green' if status=='LULUS' else 'red'}; font-weight:bold'>{status}</span>", unsafe_allow_html=True)
            
            with col_res2:
                # Visualisasi posisi nilai dalam gauge sederhana
                fig, ax = plt.subplots(figsize=(5, 1))
                plt.barh(['Math Score'], [prediction], color='teal')
                plt.xlim(0, 100)
                plt.axvline(60, color='red', linestyle='--') # Garis batas lulus
                st.pyplot(fig)
